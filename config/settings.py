"""Central configuration — v2 adds observability stack settings."""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )

    # ── AWS ───────────────────────────────────────────────────────────────────
    aws_region:  str      = "us-east-1"
    aws_profile: str | None = None

    # ── Bedrock ───────────────────────────────────────────────────────────────
    bedrock_llm_model_id:       str = "anthropic.claude-sonnet-4-5"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_backend: Literal["faiss", "qdrant", "opensearch"] = "faiss"
    faiss_index_path:     str       = "./data/faiss_index"
    qdrant_url:           str       = "http://localhost:6333"
    qdrant_collection:    str       = "incident_knowledge"
    qdrant_api_key:       str | None = None

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri:    str = "http://localhost:5000"
    mlflow_experiment_name: str = "incident-remediation"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host:         str       = "0.0.0.0"
    api_port:         int       = 8000
    api_log_level:    str       = "info"
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Agent tuning ──────────────────────────────────────────────────────────
    retriever_top_k:       int   = 5
    risk_score_threshold:  float = 0.7
    llm_max_tokens:        int   = 2048
    llm_temperature:       float = 0.2

    # ── Observability ─────────────────────────────────────────────────────────
    prometheus_port:              int       = 9090
    otel_exporter_otlp_endpoint:  str       = "http://localhost:4317"

    # Splunk HEC
    splunk_hec_url:   str | None = None
    splunk_hec_token: str | None = None

    # Honeycomb (set OTEL endpoint to https://api.honeycomb.io and add x-honeycomb-team header)
    honeycomb_api_key: str | None = None

    # Arize Phoenix
    arize_api_key:  str | None = None
    arize_space_id: str | None = None

    # ── Dev / offline ─────────────────────────────────────────────────────────
    offline_mode: bool = False

    # ── Auth ──────────────────────────────────────────────────────────────────
    # Map of API key -> caller identity, e.g. {"sk-live-abc123": "jsmith"}.
    # Empty (the default) disables auth entirely — every request resolves to
    # "anonymous". Set API_KEYS in any environment beyond local/offline dev.
    api_keys: dict[str, str] = Field(default_factory=dict)

    # Shared secret for the ServiceNow integration adapter (see
    # api/routers/servicenow.py). Placeholder auth — swap for Okta later.
    # Unset (the default) disables auth for that endpoint too, local/offline dev.
    servicenow_shared_secret: str | None = None

    @field_validator("risk_score_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("risk_score_threshold must be between 0 and 1")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
