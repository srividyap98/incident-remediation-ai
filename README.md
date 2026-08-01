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
