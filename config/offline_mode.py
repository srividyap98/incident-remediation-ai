"""
Offline / Mock Mode
-------------------
Sets up mock Bedrock responses and a pre-built FAISS index using
sentence-transformers (no AWS needed at all). Drop-in replacement
for local development without AWS credentials.

Usage — set in .env:
    OFFLINE_MODE=true

Or export before running:
    OFFLINE_MODE=true make dev

What gets mocked:
  - Bedrock LLM calls → returns realistic canned responses
  - Bedrock embeddings → uses sentence-transformers/all-MiniLM-L6-v2 (local, free)
  - No network calls leave your machine
"""
from __future__ import annotations

import json
import os
import random
from unittest.mock import MagicMock

# ── Canned LLM responses ──────────────────────────────────────────────────────

_RISK_RESPONSES = [
    {"risk_score": 0.82, "risk_factors": ["Critical service affected", "Revenue impact confirmed", "No automatic failover", "Multiple timeline events", "High blast radius"]},
    {"risk_score": 0.45, "risk_factors": ["Single service affected", "Auto-scaling available", "Low user impact", "Degraded not down", "Recent similar incident resolved quickly"]},
    {"risk_score": 0.91, "risk_factors": ["P1 severity", "Data integrity at risk", "External customer impact", "No rollback available", "Peak business hours"]},
]

_REMEDIATION_RESPONSES = [
    {
        "root_cause_analysis": "Connection pool exhaustion caused by a combination of a connection leak introduced in v3.4.1 and an unexpected 40% traffic spike. Connections were not being properly released in the new async request handler, causing the pool to fill over approximately 90 minutes.",
        "remediation_steps": [
            "Immediately restart affected service pods to release leaked connections: kubectl rollout restart deployment/payments-api",
            "Kill idle DB connections older than 10 minutes via pg_terminate_backend()",
            "Temporarily increase connection pool size: set DATABASE_POOL_SIZE=100 and redeploy",
            "Roll back to v3.3.9 to eliminate the connection leak: kubectl set image deployment/payments-api app=payments-api:v3.3.9",
            "Scale out horizontally to handle traffic: kubectl scale deployment/payments-api --replicas=8",
            "Monitor pg_stat_activity until connection count normalizes below 200",
        ],
        "confidence_score": 0.88,
        "additional_notes": "Root cause confirmed as the new async handler in v3.4.1 — review PR #847 for the specific code change. Schedule post-mortem within 48 hours.",
    },
    {
        "root_cause_analysis": "Memory leak in the order-service v3.4.1 introduced an unbounded in-memory cache in the product catalog service. The cache has no TTL or maxsize configured, growing until the 3GB container limit is hit every ~2 hours.",
        "remediation_steps": [
            "Immediately roll back to v3.3.9: kubectl set image deployment/order-service app=order-service:v3.3.9",
            "Verify rollback is healthy: kubectl rollout status deployment/order-service",
            "Capture heap dump from a running pod for investigation: kubectl exec <pod> -- jmap -dump:live,format=b,file=/tmp/heap.hprof 1",
            "In the v3.4.1 code, add maxsize=10000 and TTL=300 to the ProductCatalogCache initialization",
            "After fix, load test in staging with memory profiler enabled before re-deploying",
        ],
        "confidence_score": 0.91,
        "additional_notes": "Check ProductCatalogCache in src/cache/catalog.py — the maxsize parameter was removed in the refactor. This is a regression.",
    },
    {
        "root_cause_analysis": "Kafka consumer encountered a malformed JSON message (poison pill) at partition 3, offset 2847201. The deserializer throws an exception, the consumer retries indefinitely, and the entire partition is blocked. Other partitions continue normally.",
        "remediation_steps": [
            "Inspect the poison pill message: kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic order-events --partition 3 --offset 2847201 --max-messages 1",
            "Skip the bad message by committing offset 2847202: kafka-consumer-groups.sh --reset-offsets --to-offset 2847202 --execute --group inventory-updater --topic order-events:3",
            "Restart the consumer to resume processing: kubectl rollout restart deployment/inventory-updater",
            "Save the poison pill message to S3 for forensics: kafka-console-consumer.sh ... > /tmp/poison.json && aws s3 cp /tmp/poison.json s3://incidents/poison-pills/",
            "Add a Dead Letter Queue so future poison pills are captured instead of blocking",
            "Monitor consumer lag until it returns to <1000 messages",
        ],
        "confidence_score": 0.94,
        "additional_notes": "The malformed message likely originates from a producer that serialized a null field incorrectly. Identify the producer service from the message headers and add a null check.",
    },
]


def build_mock_bedrock_client():
    """Return a boto3 client mock that returns realistic Bedrock responses."""
    client = MagicMock()

    def converse_side_effect(**kwargs):
        prompt = ""
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            if content:
                prompt = content[0].get("text", "")

        # Route to risk or remediation response based on prompt content
        if "risk_score" in prompt.lower() or "risk analyst" in prompt.lower():
            body = json.dumps(random.choice(_RISK_RESPONSES))
        else:
            body = json.dumps(random.choice(_REMEDIATION_RESPONSES))

        return {
            "output": {"message": {"content": [{"text": body}]}},
            "usage": {"inputTokens": 500, "outputTokens": 200},
        }

    client.converse.side_effect = converse_side_effect
    return client


def build_local_embeddings():
    """
    Use sentence-transformers for local embeddings (no AWS needed).
    Install: pip install sentence-transformers
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("  Using local sentence-transformers embeddings (offline mode)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except ImportError:
        raise ImportError(
            "For offline mode, install: pip install sentence-transformers\n"
            "Or configure AWS credentials for Bedrock embeddings."
        )


def install_offline_patches():
    """
    Monkey-patch boto3 and embeddings so the app runs with zero AWS calls.
    Call this before importing any agent modules.
    """
    import boto3
    boto3.client = MagicMock(return_value=build_mock_bedrock_client())

    # Patch the embeddings factory
    import rag.embeddings.bedrock as emb_module
    emb_module.get_embeddings = build_local_embeddings

    print("  Offline mode active — all AWS calls are mocked")


# ── Auto-activate if OFFLINE_MODE=true ───────────────────────────────────────
if os.getenv("OFFLINE_MODE", "").lower() in ("true", "1", "yes"):
    install_offline_patches()
