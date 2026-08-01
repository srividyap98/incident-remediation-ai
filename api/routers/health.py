"""Health check endpoints."""
from fastapi import APIRouter
from api.schemas.models import HealthResponse
from config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store_backend=settings.vector_store_backend,
        bedrock_model=settings.bedrock_llm_model_id,
    )


@router.get("/")
async def root():
    return {"message": "Incident Remediation AI — see /docs for API reference."}
