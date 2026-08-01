"""
Dev Database Seeder
-------------------
Creates a realistic fake knowledge base for local development.
No AWS S3 needed — writes runbooks to ./data/sample_runbooks/ and
indexes them into a local FAISS vector store.

Run:
    python scripts/seed_dev_db.py

What it creates:
    data/sample_runbooks/     — 12 realistic incident runbooks (plain text)
    data/faiss_index.faiss    — FAISS vector index (embeddings via Bedrock Titan)
    data/faiss_index.pkl      — FAISS metadata store
    data/seed_incidents.json  — 6 sample incidents you can POST to the API
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config.offline_mode  # noqa: E402,F401 — must patch boto3/embeddings before rag imports

RUNBOOKS_DIR = Path("data/sample_runbooks")
SEED_INCIDENTS_PATH = Path("data/seed_incidents.json")

# ── Fake runbook content ──────────────────────────────────────────────────────
RUNBOOKS: dict[str, str] = {
    "db_connection_pool.txt": """
RUNBOOK: Database Connection Pool Exhaustion
Service: Any service using PostgreSQL or MySQL
Severity: P1 / P2

SYMPTOMS
- Application logs show "too many connections" or "connection pool exhausted"
- Health check endpoints return 503
- Database connection count at or near max_connections limit

ROOT CAUSES (most common)
1. Connection leak — connections opened but not properly closed
2. Sudden traffic spike overwhelming pool size
3. Long-running queries holding connections
4. Misconfigured pool size after a deployment

REMEDIATION STEPS
1. Immediate: Restart the affected service pods to release leaked connections
   kubectl rollout restart deployment/<service-name> -n <namespace>
2. Check active connections on the DB:
   SELECT count(*), state FROM pg_stat_activity GROUP BY state;
3. Kill idle connections older than 10 minutes:
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'idle' AND query_start < NOW() - INTERVAL '10 minutes';
4. If traffic spike: scale out the service horizontally
   kubectl scale deployment/<service-name> --replicas=<n>
5. Increase pool size in application config (DATABASE_POOL_SIZE env var)
6. Set connection timeout: DATABASE_POOL_TIMEOUT=30

POST-INCIDENT
- Review slow query log for queries >5s
- Set statement_timeout = 30000 in postgres.conf
- Add PgBouncer connection pooler if recurring

ESCALATION: DBA team if DB is unresponsive after step 3.
""",

    "kubernetes_oom.txt": """
RUNBOOK: Kubernetes OOMKilled — Pod Memory Exhaustion
Service: Any containerized workload on EKS/K8s
Severity: P2 / P3

SYMPTOMS
- Pods in CrashLoopBackOff with reason OOMKilled
- kubectl describe pod shows: Last State: Terminated, Reason: OOMKilled
- Prometheus alert: container_memory_usage_bytes exceeds limit

ROOT CAUSES
1. Memory leak in application code (most common)
2. Memory limit set too low for actual workload
3. Unexpected spike in request volume
4. Large in-memory cache growing unbounded

REMEDIATION STEPS
1. Check which container was OOMKilled:
   kubectl describe pod <pod-name> -n <namespace>
2. View memory usage trend:
   kubectl top pods -n <namespace>
3. Temporary fix — increase memory limit in deployment:
   kubectl set resources deployment/<name> --limits=memory=2Gi
4. Capture heap dump before restarting (if JVM):
   kubectl exec <pod> -- jmap -dump:live,format=b,file=/tmp/heap.hprof 1
   kubectl cp <pod>:/tmp/heap.hprof ./heap.hprof
5. Restart pods to recover service:
   kubectl rollout restart deployment/<name>
6. If recurring: add Vertical Pod Autoscaler (VPA)

INVESTIGATION
- Check for unbounded caches: grep for TTL=None or maxsize=None in code
- Review recent deployments: git log --oneline -20
- Profile memory with py-spy (Python) or async-profiler (JVM)

POST-INCIDENT
- Set proper resource requests AND limits (requests < limits)
- Enable memory alerting at 80% of limit
""",

    "api_high_latency.txt": """
RUNBOOK: API High Latency / Timeout Cascade
Service: REST APIs, GraphQL, gRPC services
Severity: P1 / P2

SYMPTOMS
- p99 latency > 5s (normal baseline < 200ms)
- Timeout errors appearing in logs: context deadline exceeded
- Downstream services showing elevated error rates
- Customers reporting slow page loads

ROOT CAUSES
1. Slow database query (most common) — missing index, table scan
2. Downstream dependency degraded (third-party API, internal service)
3. Thread/goroutine pool exhausted
4. N+1 query problem introduced in recent deploy
5. CDN misconfiguration causing cache misses

REMEDIATION STEPS
1. Identify slow queries immediately:
   SELECT query, mean_exec_time, calls FROM pg_stat_statements
   ORDER BY mean_exec_time DESC LIMIT 10;
2. Check downstream health:
   curl -w "@curl-format.txt" -s https://<dependency>/health
3. Enable request tracing to find bottleneck:
   Check Datadog APM / Jaeger for traces with high duration
4. If a specific query: add EXPLAIN ANALYZE and create index
   CREATE INDEX CONCURRENTLY idx_<table>_<column> ON <table>(<column>);
5. Temporarily increase timeout budget to prevent cascades:
   Set UPSTREAM_TIMEOUT=10s in the calling service
6. If thread pool exhausted: increase worker count or scale horizontally

QUICK WINS
- Enable database query result caching (Redis) for repeated queries
- Add circuit breaker to slow downstream calls
- Check if recent deploy changed query patterns: git diff HEAD~1 -- **/*.sql

POST-INCIDENT: Establish p99 SLO alerts and runbook links in alert bodies.
""",

    "redis_cache_failure.txt": """
RUNBOOK: Redis Cache Failure / Eviction Storm
Service: Any service using Redis for caching or sessions
Severity: P2

SYMPTOMS
- Hit rate drops from >90% to <20% suddenly
- CPU spike on application servers (cache stampede)
- Redis logs: OOM command not allowed, evicted keys count rising
- Session loss, users logged out unexpectedly

ROOT CAUSES
1. Redis maxmemory reached — keys being evicted
2. Redis instance crashed or restarted (data loss if no persistence)
3. Network partition between app and Redis
4. Incorrect TTL settings causing key buildup

REMEDIATION STEPS
1. Check Redis memory immediately:
   redis-cli INFO memory | grep used_memory_human
   redis-cli INFO stats | grep evicted_keys
2. If memory full — check for runaway keys:
   redis-cli --bigkeys
   redis-cli MEMORY USAGE <largest-key>
3. Clear problematic large keys (if safe):
   redis-cli DEL <key-name>
4. If Redis crashed: promote replica
   AWS ElastiCache: automatic failover if Multi-AZ enabled
   redis-cli CLUSTER FAILOVER (Redis Cluster mode)
5. Enable persistence to prevent future data loss:
   CONFIG SET save "60 1000"
   CONFIG SET appendonly yes
6. Add jitter to cache population to prevent stampede:
   ttl = base_ttl + random.randint(0, base_ttl // 4)

POST-INCIDENT
- Set maxmemory-policy to allkeys-lru
- Alert on memory >75% and hit rate <80%
- Review key expiry — all cached data must have a TTL
""",

    "disk_space_critical.txt": """
RUNBOOK: Disk Space Critical — Node or Container
Service: Any, especially log-heavy services and databases
Severity: P1 (if >95% full)

SYMPTOMS
- Disk usage alert > 85% / 95%
- Write errors in application logs: No space left on device
- Database write failures
- Log rotation stopped

REMEDIATION STEPS
1. Identify largest directories immediately:
   du -sh /* 2>/dev/null | sort -rh | head -20
   du -sh /var/log/* | sort -rh | head -10
2. Truncate or rotate large log files (safe):
   find /var/log -name "*.log" -mtime +7 -delete
   journalctl --vacuum-size=500M
3. Clean Docker artifacts if on a K8s node:
   docker system prune -f
   crictl rmi --prune (containerd)
4. If database data directory full:
   - Run VACUUM FULL on PostgreSQL (creates temp space first — ensure 2x free space)
   - Delete old partitions if using table partitioning
5. Expand disk (AWS EBS):
   aws ec2 modify-volume --volume-id vol-xxxx --size <new-size-gb>
   Then: resize2fs /dev/xvda1

PREVENTION
- Set up log rotation: /etc/logrotate.d/<service>
- Kubernetes: configure log rotation in kubelet (containerLogMaxSize: 50Mi)
- Alert at 75% and 90% — 95% is too late
""",

    "ssl_cert_expiry.txt": """
RUNBOOK: SSL Certificate Expiry
Service: Any HTTPS endpoint
Severity: P1 if expired, P3 if <14 days remaining

SYMPTOMS
- Browser shows NET::ERR_CERT_DATE_INVALID
- curl: SSL certificate problem: certificate has expired
- Monitoring alert: cert_expiry_days < 14

REMEDIATION STEPS
1. Check certificate expiry:
   echo | openssl s_client -connect <domain>:443 2>/dev/null | openssl x509 -noout -dates
2. If using AWS Certificate Manager (ACM):
   - ACM auto-renews — check why renewal failed
   - aws acm describe-certificate --certificate-arn <arn>
   - Common cause: DNS validation CNAME record missing
3. If using Let's Encrypt / certbot:
   certbot renew --dry-run
   certbot renew --force-renewal
   systemctl reload nginx
4. If manual cert: generate CSR and submit to CA immediately
5. Update load balancer / CDN with new certificate ARN
6. Verify with: curl -v https://<domain> 2>&1 | grep -E "expire|valid"

POST-INCIDENT
- Set calendar reminder 60 days before expiry
- Enable ACM auto-renewal and DNS validation
- Add monitoring: check cert expiry weekly
""",

    "kafka_consumer_lag.txt": """
RUNBOOK: Kafka Consumer Lag — Message Processing Backlog
Service: Event-driven services, stream processors
Severity: P2

SYMPTOMS
- Consumer lag metric > 10,000 messages
- Event processing delayed by minutes/hours
- Downstream data pipelines stale
- Alert: kafka_consumer_group_lag > threshold

ROOT CAUSES
1. Consumer is too slow (processing bottleneck)
2. Kafka partition count too low for throughput
3. Consumer crashed / not running
4. Poison pill message causing repeated failures

REMEDIATION STEPS
1. Check consumer group status:
   kafka-consumer-groups.sh --bootstrap-server <broker> --describe --group <group>
2. If consumer is down: restart it
   kubectl rollout restart deployment/<consumer-service>
3. If poison pill (same offset retried repeatedly):
   - Identify the bad message: kafka-console-consumer.sh --offset <n> --max-messages 1
   - Skip it: manually commit the offset past the bad message
   kafka-consumer-groups.sh --reset-offsets --to-offset <n+1> --execute --group <g> --topic <t>
4. If processing is too slow: scale out consumers (up to partition count)
   kubectl scale deployment/<consumer> --replicas=<partition-count>
5. Increase partitions if needed (cannot decrease):
   kafka-topics.sh --alter --topic <topic> --partitions <n>

POST-INCIDENT
- Set Dead Letter Queue (DLQ) for unparseable messages
- Alert at lag > 1000 with 5-min window
""",

    "aws_bedrock_throttling.txt": """
RUNBOOK: AWS Bedrock Throttling / Rate Limit Errors
Service: incident-remediation-ai — LLM inference layer
Severity: P2 / P3

SYMPTOMS
- Bedrock API returns ThrottlingException or ServiceQuotaExceededException
- Agent workflow errors: "Rate exceeded" in logs
- LLM response latency > 30s before timeout

ROOT CAUSES
1. Exceeded on-demand throughput quota (tokens per minute)
2. Concurrent request limit hit
3. Account-level quota not increased for production load

REMEDIATION STEPS
1. Check current quota:
   aws service-quotas get-service-quota --service-code bedrock \
     --quota-code L-<quota-id> --region us-east-1
2. Immediate: enable exponential backoff (already implemented in response_generator.py via tenacity)
   Verify RETRY_MAX_ATTEMPTS=3, RETRY_WAIT_EXPONENTIAL_MAX=15 in .env
3. Request quota increase:
   AWS Console → Service Quotas → Amazon Bedrock → Request increase
   (Takes 1-3 business days)
4. Implement request queuing to smooth burst traffic:
   Add SQS queue in front of LLM calls, process at steady rate
5. Enable Provisioned Throughput for predictable high-volume workloads:
   aws bedrock create-provisioned-model-throughput \
     --model-id anthropic.claude-sonnet-4-5 \
     --provisioned-model-name prod-throughput \
     --model-units 1

COST NOTE: Provisioned throughput is ~$40/hour per model unit. Only enable for sustained load >80% utilisation.

POST-INCIDENT
- Add CloudWatch alarm on Bedrock ThrottledRequests metric
- Implement request deduplication for identical incident queries
""",

    "faiss_index_corruption.txt": """
RUNBOOK: FAISS Index Corruption or Stale Embeddings
Service: incident-remediation-ai — RAG retrieval layer
Severity: P2

SYMPTOMS
- Retrieval returns empty results or clearly irrelevant documents
- Error: RuntimeError: Error in faiss, or index.ntotal returns 0
- Confidence scores consistently below 0.3
- Drift detector alert: retrieval_drift_score > 0.25

ROOT CAUSES
1. Incomplete write during ingestion (power loss, OOM during indexing)
2. Index file corrupted during container restart
3. Embeddings model changed without re-indexing
4. Knowledge base is empty / ingestion was never run

REMEDIATION STEPS
1. Verify index health:
   python -c "
   import faiss, pickle
   idx = faiss.read_index('data/faiss_index.faiss')
   print('Vectors in index:', idx.ntotal)
   "
2. If empty or corrupt — rebuild from source:
   python -m rag.ingestion.pipeline --source ./data/sample_runbooks/
3. Validate after rebuild:
   python -c "
   from rag.vector_store.factory import get_vector_store
   store = get_vector_store()
   results = store.similarity_search('database connection timeout', k=3)
   print('Top result:', results[0].page_content[:100])
   "
4. If using wrong embeddings model: delete old index and re-run ingestion
   rm data/faiss_index.faiss data/faiss_index.pkl
   python -m rag.ingestion.pipeline --source ./data/sample_runbooks/
5. Save a backup after each successful ingestion:
   cp data/faiss_index.faiss data/faiss_index.faiss.bak

POST-INCIDENT
- Run ingestion pipeline in Airflow with index validation task (already in ingestion_dag.py)
- Store FAISS index in S3 with versioning enabled for production
""",

    "network_partition.txt": """
RUNBOOK: Network Partition Between Services
Service: Any microservices communicating over internal network
Severity: P1

SYMPTOMS
- Sudden spike in connection timeout errors between specific service pair
- One-directional failure (A can reach B but B cannot reach A)
- AWS: VPC Flow Logs showing REJECT entries
- Network policy or security group recently changed

REMEDIATION STEPS
1. Confirm the partition is real (not DNS):
   kubectl exec -it <pod-a> -- curl -v http://<service-b>/health
   kubectl exec -it <pod-b> -- curl -v http://<service-a>/health
2. Check DNS resolution:
   kubectl exec -it <pod> -- nslookup <service-name>.<namespace>.svc.cluster.local
3. Inspect NetworkPolicy (Kubernetes):
   kubectl get networkpolicy -n <namespace>
   kubectl describe networkpolicy <policy-name>
4. Check AWS Security Groups:
   aws ec2 describe-security-groups --group-ids <sg-id>
   Ensure inbound rule allows traffic from source SG on the correct port
5. Temporary fix — open the blocked port (document and review later):
   aws ec2 authorize-security-group-ingress \
     --group-id <sg-id> --protocol tcp --port <port> --source-group <source-sg>
6. If NACl blocking: check subnet ACLs (stateless, must allow return traffic)

ROOT CAUSE ANALYSIS
- Review recent Terraform / CloudFormation changes: git log --oneline -10 -- infra/
- Check if a new NetworkPolicy was applied via CI/CD

POST-INCIDENT
- Add integration test that verifies service connectivity on every deploy
- Document all required ports in infra/network_policy_map.md
""",

    "mlflow_tracking_unavailable.txt": """
RUNBOOK: MLflow Tracking Server Unavailable
Service: incident-remediation-ai — experiment tracking
Severity: P3

SYMPTOMS
- MLflow UI (localhost:5000) not loading
- Python logs: MlflowException: Could not connect to tracking server
- Airflow ingestion DAG failing at mlflow.start_run() step

NOTE: MLflow tracking failure does NOT stop incident remediation.
      The workflow degrades gracefully — tracking is non-critical path.

REMEDIATION STEPS
1. Check if MLflow container is running (local dev):
   docker ps | grep mlflow
   docker compose logs mlflow
2. Restart MLflow:
   docker compose restart mlflow
3. If DB backend (SQLite) locked:
   docker compose exec mlflow rm -f /mlflow/mlflow.db-shm /mlflow/mlflow.db-wal
   docker compose restart mlflow
4. If artifact storage full:
   docker compose exec mlflow du -sh /mlflow/artifacts
   docker compose exec mlflow find /mlflow/artifacts -mtime +30 -delete
5. Verify config in .env:
   MLFLOW_TRACKING_URI=http://localhost:5000
   MLFLOW_EXPERIMENT_NAME=incident-remediation

PRODUCTION (AWS)
- Use RDS PostgreSQL as backend store instead of SQLite
- Store artifacts in S3: mlflow server --default-artifact-root s3://your-bucket/mlflow/
""",

    "general_incident_template.txt": """
GENERAL INCIDENT RESPONSE TEMPLATE
For incidents not covered by a specific runbook.

IMMEDIATE RESPONSE (first 5 minutes)
1. Acknowledge the alert — assign an incident commander
2. Assess blast radius: how many users/services affected?
3. Create incident channel: #inc-YYYY-MM-DD-<short-description>
4. Post initial status update to status page

INVESTIGATION CHECKLIST
- What changed in the last 24 hours? (deploys, config changes, data migrations)
  git log --since="24 hours ago" --all --oneline
- Is it isolated to one region/AZ/pod or widespread?
- Are there correlated alerts firing simultaneously?
- Check error rate, latency, and saturation (USE method)

COMMUNICATION CADENCE
- P1: Update every 15 minutes until resolved
- P2: Update every 30 minutes
- P3/P4: Update at start and resolution

RESOLUTION
1. Apply fix with rollback plan ready
2. Verify with synthetic tests before declaring resolved
3. Monitor for 30 minutes post-fix

POST-INCIDENT (within 48 hours)
1. Write incident report: timeline, root cause, impact, action items
2. Create runbook if one doesn't exist
3. Add automated test to prevent regression
4. Schedule blameless post-mortem if P1/P2

ESCALATION PATH
- Engineering on-call → Team lead → Director of Engineering → CTO
""",
}

# ── Sample incidents for API testing ─────────────────────────────────────────
SEED_INCIDENTS = [
    {
        "incident_id": "INC-2024-001",
        "incident_title": "Payments API — database connection pool exhausted",
        "incident_description": (
            "All connections to the primary PostgreSQL instance are being refused. "
            "The payments-api service is returning 503 to all checkout requests. "
            "Connection count shows 500/500 active connections. Revenue impact confirmed."
        ),
        "severity": "P1",
        "affected_service": "payments-api",
        "timeline_events": [
            {"timestamp": "2024-11-01T09:00:00Z", "event": "PagerDuty alert fired: DB connections > 480", "source": "pagerduty"},
            {"timestamp": "2024-11-01T09:02:00Z", "event": "Payments returning 503", "source": "datadog"},
            {"timestamp": "2024-11-01T09:04:00Z", "event": "On-call engineer engaged", "source": "slack"},
        ],
        "raw_logs": [
            "ERROR: remaining connection slots are reserved for non-replication superuser connections",
            "FATAL: sorry, too many clients already",
            "ERROR: connection pool exhausted after 30s wait",
            "WARN: health check failed: could not connect to server",
        ],
    },
    {
        "incident_id": "INC-2024-002",
        "incident_title": "Order service pods OOMKilled — memory leak",
        "incident_description": (
            "order-service pods are repeatedly OOMKilled every 2 hours. "
            "Each pod uses 2.8GB before hitting the 3GB limit. "
            "Began after the v3.4.1 deployment yesterday at 14:00 UTC."
        ),
        "severity": "P2",
        "affected_service": "order-service",
        "timeline_events": [
            {"timestamp": "2024-11-02T06:00:00Z", "event": "v3.4.1 deployed", "source": "ci-cd"},
            {"timestamp": "2024-11-02T08:05:00Z", "event": "First OOMKill alert", "source": "prometheus"},
            {"timestamp": "2024-11-02T10:10:00Z", "event": "Second OOMKill cycle", "source": "prometheus"},
        ],
        "raw_logs": [
            "OOMKilled: container order-service exceeded memory limit 3Gi",
            "GC overhead limit exceeded",
            "java.lang.OutOfMemoryError: Java heap space",
            "WARN: Large object cache: 248,000 entries, estimated 2.1GB",
        ],
    },
    {
        "incident_id": "INC-2024-003",
        "incident_title": "Search API p99 latency spiked to 12s",
        "incident_description": (
            "Search endpoint latency degraded from 180ms to 12s at p99. "
            "Affecting approximately 30% of search requests. "
            "Elasticsearch cluster shows green health but query times are high."
        ),
        "severity": "P2",
        "affected_service": "search-api",
        "timeline_events": [
            {"timestamp": "2024-11-03T14:00:00Z", "event": "Latency alert: p99 > 5s", "source": "datadog"},
            {"timestamp": "2024-11-03T14:05:00Z", "event": "Customer complaints on Twitter", "source": "social"},
        ],
        "raw_logs": [
            "WARN: Elasticsearch query took 11432ms for query: {match_all: {}}",
            "ERROR: Request timeout after 12000ms on /search?q=laptop",
            "INFO: Elasticsearch heap usage: 28GB / 32GB",
        ],
    },
    {
        "incident_id": "INC-2024-004",
        "incident_title": "Redis session cache evicting keys — users being logged out",
        "incident_description": (
            "Users are reporting being randomly logged out mid-session. "
            "Redis memory is at 99% and eviction rate is 50k keys/second. "
            "Session keys appear to be evicted before TTL expiry."
        ),
        "severity": "P2",
        "affected_service": "auth-service",
        "timeline_events": [
            {"timestamp": "2024-11-04T11:00:00Z", "event": "User complaints via support", "source": "zendesk"},
            {"timestamp": "2024-11-04T11:20:00Z", "event": "Redis memory alert 95%", "source": "prometheus"},
        ],
        "raw_logs": [
            "WARN: Redis evicted_keys: 52341 (last 60s)",
            "ERROR: Session not found for token abc123 — user forced to re-login",
            "INFO: Redis used_memory: 15.8gb / 16gb",
            "WARN: Redis hit_rate dropped from 94% to 18%",
        ],
    },
    {
        "incident_id": "INC-2024-005",
        "incident_title": "SSL certificate expired on api.company.com",
        "incident_description": (
            "The SSL certificate for api.company.com expired at 00:00 UTC today. "
            "All HTTPS traffic is being rejected with certificate error. "
            "Mobile app and third-party integrations are fully broken."
        ),
        "severity": "P1",
        "affected_service": "api-gateway",
        "timeline_events": [
            {"timestamp": "2024-11-05T00:01:00Z", "event": "Certificate expired", "source": "system"},
            {"timestamp": "2024-11-05T00:05:00Z", "event": "Synthetic monitor alert fired", "source": "datadog"},
            {"timestamp": "2024-11-05T06:30:00Z", "event": "On-call engineer woken up", "source": "pagerduty"},
        ],
        "raw_logs": [
            "SSL_ERROR_RX_RECORD_TOO_LONG",
            "curl: (60) SSL certificate problem: certificate has expired",
            "Error: self-signed certificate in certificate chain",
        ],
    },
    {
        "incident_id": "INC-2024-006",
        "incident_title": "Kafka consumer lag — event processing 4 hours behind",
        "incident_description": (
            "The inventory-updater consumer group is 4 hours behind on the order-events topic. "
            "Product inventory counts in the warehouse system are stale. "
            "One partition appears stuck at offset 2847201 — not advancing."
        ),
        "severity": "P2",
        "affected_service": "inventory-service",
        "timeline_events": [
            {"timestamp": "2024-11-06T08:00:00Z", "event": "Consumer lag alert: 200k messages", "source": "prometheus"},
            {"timestamp": "2024-11-06T10:00:00Z", "event": "Lag grew to 800k messages", "source": "prometheus"},
            {"timestamp": "2024-11-06T12:00:00Z", "event": "Inventory data 4h stale confirmed", "source": "ops"},
        ],
        "raw_logs": [
            "ERROR: Failed to deserialize message at offset 2847201: JsonParseException",
            "WARN: Consumer group inventory-updater, partition 3: lag=847201",
            "ERROR: Caused by: com.fasterxml.jackson.core.JsonParseException: Unexpected character",
            "INFO: Retrying message at offset 2847201 (attempt 847 of max 1000)",
        ],
    },
]


# ── Main seeder ───────────────────────────────────────────────────────────────

def write_runbooks() -> None:
    RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in RUNBOOKS.items():
        path = RUNBOOKS_DIR / filename
        path.write_text(content.strip())
    print(f"  Wrote {len(RUNBOOKS)} runbooks to {RUNBOOKS_DIR}/")


def write_seed_incidents() -> None:
    SEED_INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_INCIDENTS_PATH.write_text(json.dumps(SEED_INCIDENTS, indent=2))
    print(f"  Wrote {len(SEED_INCIDENTS)} sample incidents to {SEED_INCIDENTS_PATH}")


def build_faiss_index() -> None:
    print("  Building FAISS index via Bedrock Titan embeddings...")
    print("  (This calls AWS Bedrock — ensure AWS credentials are configured)")
    from rag.ingestion.pipeline import run_ingestion
    n = run_ingestion(str(RUNBOOKS_DIR))
    print(f"  Indexed {n} chunks into FAISS at data/faiss_index")


def print_next_steps() -> None:
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dev database seeded successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:

1. Start the API:
   make dev

2. Open the Swagger UI:
   http://localhost:8000/docs

3. Test with a sample incident (pick any from data/seed_incidents.json):
   python scripts/test_api.py

4. Or use curl directly:
   curl -X POST http://localhost:8000/api/v1/incidents/ \\
     -H "Content-Type: application/json" \\
     -d @data/seed_incidents.json  # sends first incident

5. View MLflow experiment results:
   make mlflow-ui  →  http://localhost:5000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    print("\nSeeding dev database...")
    print()

    print("[1/3] Writing runbooks...")
    write_runbooks()

    print("[2/3] Writing sample incidents...")
    write_seed_incidents()

    print("[3/3] Building FAISS vector index...")
    try:
        build_faiss_index()
    except Exception as e:
        print(f"\n  WARNING: Could not build FAISS index: {e}")
        print("  Make sure AWS credentials are configured: aws configure")
        print("  Then re-run: python scripts/seed_dev_db.py")
        print("  Runbooks and sample incidents were still written successfully.\n")
        sys.exit(0)

    print_next_steps()
