# Incident Remediation AI

> Multi-Agent GenAI System for Incident Remediation & Risk Intelligence
> Built on LangGraph + AWS Bedrock (Claude Sonnet 4.5)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Data Ingestion Layer                                        │
│  PySpark · Airflow · MLflow · S3                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  RAG Pipeline — AWS Bedrock                                  │
│  Titan Embeddings → FAISS / Qdrant Vector Store             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Multi-Agent Orchestration — LangGraph                       │
│  [Retriever] → [ContextBuilder] → [RiskEvaluator]           │
│                                         │                    │
│                          ┌──────────────┴──────────────┐    │
│                   score < 0.7                    score ≥ 0.7 │
│                          │                            │      │
│               [ResponseGenerator]            [HITL Review]  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  FastAPI Inference API — AWS EKS                             │
│  /api/v1/incidents  ·  /api/v1/hitl/resume  ·  /metrics     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Monitoring & Feedback                                       │
│  Prometheus · Grafana · MLflow · Drift Detector             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
incident-remediation-ai/
├── agents/
│   ├── state.py                   # Shared LangGraph IncidentState
│   ├── retriever/agent.py         # Semantic search against vector store
│   ├── context_builder/agent.py   # MCP-style context assembly
│   ├── risk_evaluator/agent.py    # Heuristic + LLM risk scoring
│   ├── response_generator/agent.py# Claude Sonnet 4.5 remediation gen
│   └── orchestrator/graph.py      # LangGraph StateGraph + HITL routing
├── rag/
│   ├── embeddings/bedrock.py      # Bedrock Titan embeddings
│   ├── vector_store/factory.py    # FAISS / Qdrant swappable backend
│   └── ingestion/pipeline.py      # S3/local → chunk → embed → upsert
├── api/
│   ├── main.py                    # FastAPI app, middleware, routers
│   ├── routers/incidents.py       # POST /api/v1/incidents
│   ├── routers/hitl.py            # POST /api/v1/hitl/resume
│   ├── routers/health.py          # GET /health
│   └── schemas/models.py          # Pydantic request/response models
├── pipelines/
│   ├── airflow_dags/ingestion_dag.py  # Nightly knowledge ingestion DAG
│   └── spark_jobs/log_preprocessor.py # PySpark log cleaning job
├── monitoring/
│   ├── metrics.py                 # Prometheus metric registry
│   ├── mlflow_tracking.py         # MLflow experiment utilities
│   ├── drift_detector.py          # Embedding + confidence drift
│   └── dashboards/prometheus.yml  # Scrape config
├── infra/
│   ├── eks/deployment.yaml        # Kubernetes manifests (EKS)
│   ├── lambda/lambda_handler.py   # Async Lambda processor
│   └── terraform/main.tf          # S3, DynamoDB, SQS, IAM, SNS
├── tests/
│   ├── unit/                      # Agent-level unit tests (mocked)
│   └── integration/               # Full workflow integration tests
├── config/settings.py             # Pydantic-settings config loader
├── Dockerfile                     # Multi-stage production image
├── docker-compose.yml             # Full local dev stack
└── Makefile                       # Developer commands
```

## Request Lifecycle — Step by Step

This walks through everything that happens to **one** `POST /api/v1/incidents/` request, from the moment it hits the process to the moment a response goes back over the wire. Each step names the exact file responsible.

```
Client
  │  POST /api/v1/incidents/
  ▼
[0] logging_middleware ───────────────── api/main.py
  ▼
[1] get_current_principal (auth) ─────── api/auth.py
  ▼
[2] submit_incident (router) ─────────── api/routers/incidents.py
  ▼
[3] get_graph().stream(...) ──────────── agents/orchestrator/graph.py
  │
  ├─[4] retriever ─────────────────────── agents/retriever/agent.py
  │       └─ get_vector_store() ───────── rag/vector_store/factory.py
  │             └─ get_embeddings() ───── rag/embeddings/bedrock.py  (Titan v2, AWS Bedrock)
  │             └─ FAISS or Qdrant similarity search
  │
  ├─[5] context_builder ───────────────── agents/context_builder/agent.py
  │
  ├─[6] risk_evaluator ─────────────────── agents/risk_evaluator/agent.py
  │       └─ boto3 bedrock-runtime.converse()  (Claude Sonnet 4.5)
  │
  ├─[7] route_after_risk ──────────────── agents/orchestrator/graph.py
  │       ├─ score < threshold ─────────┐
  │       └─ score ≥ threshold ─────┐   │
  │                                 ▼   │
  │            [7a] human_review_node   │  agents/orchestrator/graph.py
  │                  interrupt() → graph pauses, checkpointed by MemorySaver
  │                  → API returns status=pending_review immediately
  │                  ... (later) POST /api/v1/hitl/resume ─ api/routers/hitl.py
  │                  → Command(resume=...) re-enters the graph at this node
  │                                 │   │
  │                                 ▼   ▼
  ├─[8] response_generator ────────────── agents/response_generator/agent.py
  │       └─ boto3 bedrock-runtime.converse()  (Claude Sonnet 4.5, JSON mode)
  │
  ├─[9] evaluator ──────────────────────── agents/orchestrator/graph.py (evaluator_node)
  │       ├─ check_grounding() ─────────── evaluation/hallucination_guard.py
  │       └─ log_llm_interaction() ─────── monitoring/arize_logger.py
  │
  ▼
[10] END → final_state returned to router
  ▼
[11] IncidentResponse built ──────────── api/routers/incidents.py, api/schemas/models.py
  ▼
[12] Prometheus metrics + response log ─ api/main.py (logging_middleware)
  ▼
Client ◄── JSON response
```

### Step-by-step detail

0. **Request enters the process** — [api/main.py](api/main.py)'s `logging_middleware` fires first for every request: generates a `request_id` (UUID), binds it to `structlog`'s contextvars so every downstream log line carries it, and starts a latency timer.
1. **Authentication** — [api/auth.py](api/auth.py)'s `get_current_principal` runs as a router dependency, reading the `X-API-Key` header. If `API_KEYS` is unset (local/offline dev), it's a no-op that returns `"anonymous"`; otherwise an invalid/missing key raises `401`. This is the documented swap-in point for Okta.
2. **Routing + request validation** — FastAPI matches `POST /api/v1/incidents/` to `submit_incident` in [api/routers/incidents.py](api/routers/incidents.py), validating the body against the `IncidentRequest` Pydantic model ([api/schemas/models.py](api/schemas/models.py)). It immediately delegates to `_run_workflow_and_build_response`, the single shared entry point into the AI workflow (also reused by the ServiceNow adapter in `api/routers/servicenow.py`).
3. **Graph kickoff** — a fresh `thread_id` (UUID) is minted, the incident payload is flattened into a dict, and `get_graph()` (a process-wide cached `CompiledStateGraph`) is streamed with that input — [agents/orchestrator/graph.py](agents/orchestrator/graph.py). The `thread_id` is the LangGraph checkpoint key that makes HITL resume possible later.
4. **Retriever node** — [agents/retriever/agent.py](agents/retriever/agent.py) builds a semantic query from the incident's title/severity/service/description, then calls `get_vector_store()` ([rag/vector_store/factory.py](rag/vector_store/factory.py)), which lazily builds a FAISS or Qdrant store depending on `VECTOR_STORE_BACKEND`, embedding the query via Titan v2 ([rag/embeddings/bedrock.py](rag/embeddings/bedrock.py)). It oversamples child chunks, deduplicates by `parent_id`, and returns full parent sections (see the hierarchical chunking design in [rag/ingestion/pipeline.py](rag/ingestion/pipeline.py)) as `retrieved_docs`.
5. **Context builder node** — [agents/context_builder/agent.py](agents/context_builder/agent.py) assembles `retrieved_docs` + structured incident metadata + raw logs into one token-budgeted `enriched_context` string, ready for LLM prompts.
6. **Risk evaluator node** — [agents/risk_evaluator/agent.py](agents/risk_evaluator/agent.py) computes a heuristic score from severity/timeline, then calls Bedrock Claude directly via `boto3` (`bedrock-runtime.converse`) for an LLM-based risk score + top risk factors, blending the two (30/70). Writes `risk_score`, `risk_factors`, `requires_human_review` to state.
7. **Conditional routing** — `route_after_risk` in [agents/orchestrator/graph.py](agents/orchestrator/graph.py) branches on `requires_human_review` (i.e. `risk_score >= RISK_SCORE_THRESHOLD`):
   - **Below threshold** → straight to `response_generator`.
   - **At/above threshold** → `human_review_node` calls LangGraph's `interrupt()`, which pauses the graph and persists its state via the `MemorySaver` checkpointer, keyed by `thread_id`. The API call returns right away with `status="pending_review"` and the `thread_id`. A human later calls `POST /api/v1/hitl/resume` ([api/routers/hitl.py](api/routers/hitl.py)) with an approve/reject decision; that re-enters the *same* graph at the checkpoint and continues.
8. **Response generator node** — [agents/response_generator/agent.py](agents/response_generator/agent.py) sends `enriched_context` to Bedrock Claude Sonnet 4.5 (`converse`, JSON-mode prompt), parses the structured JSON (root cause, remediation steps, confidence), and composes the human-readable `final_response` string.
9. **Evaluator node (post-generation quality gate)** — `evaluator_node` in [agents/orchestrator/graph.py](agents/orchestrator/graph.py) runs a hallucination/grounding check via [evaluation/hallucination_guard.py](evaluation/hallucination_guard.py) (are the generated claims actually supported by `retrieved_docs`?) and logs the full interaction — prompt, response, scores, grounding warnings — to Arize via [monitoring/arize_logger.py](monitoring/arize_logger.py). Skipped if HITL rejected the incident (no `final_response` to check).
10. **Graph reaches `END`** — the last streamed state (`final_state`) is what the router receives back.
11. **Response assembly** — back in [api/routers/incidents.py](api/routers/incidents.py), `_run_workflow_and_build_response` derives `status` (`completed` / `pending_review` / `error`) from `final_state` and builds the `IncidentResponse` model ([api/schemas/models.py](api/schemas/models.py)). The `thread_id` is also stashed in the in-memory `_thread_store` so `GET /api/v1/incidents/{id}` can look up status later.
12. **Response leaves the process** — control returns through `logging_middleware` ([api/main.py](api/main.py)), which records the total duration, increments the `api_requests_total` Prometheus counter and `api_request_duration_seconds` histogram (scraped at `/metrics`), and logs the final structured log line before the JSON response is sent to the client.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS account with Bedrock access (Claude Sonnet 4.5 + Titan Embeddings enabled)

### 2. Install

```bash
git clone <repo>
cd incident-remediation-ai
cp .env.example .env          # Edit with your AWS credentials
make install
```

### 3. Run locally (FAISS + mock AWS)

```bash
# Index sample runbooks
make ingest-local

# Start API
make dev
# → http://localhost:8000/docs
```

### 4. Run full stack with Docker

```bash
make dev-docker
# API:      http://localhost:8000/docs
# MLflow:   http://localhost:5000
# Grafana:  http://localhost:3000
# Prometheus: http://localhost:9090
```

### 5. Switch to Qdrant

```bash
# In .env:
VECTOR_STORE_BACKEND=qdrant

# Start Qdrant alongside the stack:
make docker-qdrant
```

## API Usage

### Submit an incident

```bash
curl -X POST http://localhost:8000/api/v1/incidents/ \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "INC-2024-001",
    "incident_title": "Payment database connection pool exhausted",
    "incident_description": "All DB connections refused. Payments failing. Revenue impact.",
    "severity": "P1",
    "affected_service": "payments-api",
    "timeline_events": [
      {"timestamp": "2024-11-01T09:00:00Z", "event": "Alerts fired", "source": "pagerduty"}
    ],
    "raw_logs": ["ERROR: too many connections", "FATAL: connection pool exhausted"]
  }'
```

**Response (completed)**:
```json
{
  "incident_id": "INC-2024-001",
  "status": "completed",
  "risk_score": 0.847,
  "requires_human_review": false,
  "root_cause_analysis": "Connection pool exhausted due to ...",
  "remediation_steps": ["Step 1...", "Step 2..."],
  "confidence_score": 0.91,
  "risk_factors": ["High severity", "Revenue impact"]
}
```

**Response (HITL triggered)**:
```json
{
  "status": "pending_review",
  "risk_score": 0.92,
  "requires_human_review": true
}
```

### Resume after HITL

```bash
curl -X POST http://localhost:8000/api/v1/hitl/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "<thread_id_from_original_response>",
    "approved": true,
    "reviewer_id": "jsmith",
    "review_notes": "Verified. Proceed with remediation."
  }'
```

## Configuration

All configuration is via environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss` \| `qdrant` \| `opensearch` |
| `BEDROCK_LLM_MODEL_ID` | `anthropic.claude-sonnet-4-5` | Bedrock model ID |
| `RISK_SCORE_THRESHOLD` | `0.7` | Score above which HITL is triggered |
| `RETRIEVER_TOP_K` | `5` | Number of docs returned by vector search |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic responses |

## Testing

```bash
make test           # All tests + coverage report
make test-unit      # Unit tests only (no AWS needed)
make test-integration
```

## Deployment

### EKS

```bash
# Build and push image
docker build -t <ECR_URI>:latest .
docker push <ECR_URI>:latest

# Deploy
kubectl apply -f infra/eks/deployment.yaml
```

### AWS Infrastructure (Terraform)

```bash
make tf-init
make tf-plan
make tf-apply
```

## Extending the System

**Add a new agent**: Create `agents/my_agent/agent.py` with a function `my_agent(state: IncidentState) -> dict`, then register it as a node in `agents/orchestrator/graph.py`.

**Swap vector store**: Set `VECTOR_STORE_BACKEND=qdrant` in `.env` — no code changes needed.

**Add a new LLM**: Update `BEDROCK_LLM_MODEL_ID` to any Bedrock-supported model ID.

**Ingest new knowledge**: Drop `.txt` files into `./data/sample_runbooks/` and run `make ingest-local`, or point to an S3 prefix with `make ingest-s3 BUCKET=my-bucket`.

## License

MIT
