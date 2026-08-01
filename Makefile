.PHONY: install dev test lint format docker-build docker-up ingest regression feedback help

install:
	pip install -e ".[dev]"

install-observability:
	pip install -e ".[dev,observability]"

install-pipelines:
	pip install -e ".[dev,pipelines]"

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-docker:
	docker compose up --build api mlflow prometheus grafana

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/unit tests/integration -v --cov=agents --cov=rag --cov=api --cov=evaluation \
	       --cov-report=term-missing --cov-report=html:htmlcov

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

# ── Evaluation ────────────────────────────────────────────────────────────────
regression:
	python -m evaluation.prompt_regression

regression-save-baseline:
	python -m evaluation.prompt_regression --save-baseline

feedback-report:
	python -m evaluation.feedback_loop

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check .
	mypy agents/ api/ rag/ config/ monitoring/ evaluation/

format:
	ruff format .
	ruff check --fix .

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t incident-ai:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-qdrant:
	docker compose --profile qdrant up -d

# ── Data pipelines ────────────────────────────────────────────────────────────
ingest-local:
	python -m rag.ingestion.pipeline --source ./data/sample_runbooks/

ingest-s3:
	python -m rag.ingestion.pipeline --source s3://$(BUCKET)/runbooks/

seed:
	python scripts/seed_dev_db.py

# ── Infrastructure ────────────────────────────────────────────────────────────
tf-init:
	cd infra/terraform && terraform init

tf-plan:
	cd infra/terraform && terraform plan

tf-apply:
	cd infra/terraform && terraform apply -auto-approve

# ── Observability ─────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  install               Install all Python dependencies"
	@echo "  install-observability Install + Arize/Honeycomb extras"
	@echo "  dev                   Run FastAPI dev server (hot-reload)"
	@echo "  dev-docker            Run full stack via docker compose"
	@echo "  test                  Run all tests with coverage"
	@echo "  regression            Run prompt regression benchmarks"
	@echo "  feedback-report       Analyse collected feedback"
	@echo "  seed                  Seed dev database (runbooks + FAISS)"
	@echo "  ingest-local          Index sample runbooks into vector store"
	@echo "  ingest-s3             Index from S3 (set BUCKET=...)"
	@echo "  tf-plan               Terraform plan for AWS infra"
	@echo "  mlflow-ui             Launch MLflow tracking UI"
	@echo ""
