"""
AWS Bedrock Titan Embeddings
----------------------------
Wraps BedrockEmbeddings from langchain-aws with caching and logging.
"""
from __future__ import annotations

from functools import lru_cache

import structlog
from langchain_aws import BedrockEmbeddings

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_embeddings() -> BedrockEmbeddings:
    """Return a cached BedrockEmbeddings instance (Titan v2)."""
    logger.info(
        "Initialising Bedrock embeddings",
        model=settings.bedrock_embedding_model_id,
        region=settings.aws_region,
    )
    return BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )
