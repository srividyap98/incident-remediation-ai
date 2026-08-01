"""
Vector Store Factory
--------------------
Returns a LangChain-compatible VectorStore based on VECTOR_STORE_BACKEND setting.
Supports FAISS (local/dev) and Qdrant (production).

Usage:
    store = get_vector_store()
    results = store.similarity_search_with_score("database timeout", k=5)
"""
from __future__ import annotations

from functools import lru_cache

import structlog

from config.settings import get_settings
from rag.embeddings.bedrock import get_embeddings

logger = structlog.get_logger(__name__)
settings = get_settings()


def _build_faiss_store():
    import os
    from langchain_community.vectorstores import FAISS

    index_path = settings.faiss_index_path

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        logger.info("Loading existing FAISS index", path=index_path)
        return FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    logger.warning("No FAISS index found — creating empty index", path=index_path)
    # Bootstrap with a placeholder so the store is usable immediately
    store = FAISS.from_texts(
        texts=["Placeholder — index is empty. Run the ingestion pipeline."],
        embedding=get_embeddings(),
        metadatas=[{"source": "system"}],
    )
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    store.save_local(index_path)
    return store


def _build_qdrant_store():
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("Connecting to Qdrant", url=settings.qdrant_url, collection=settings.qdrant_collection)

    embeddings = get_embeddings()

    if not client.collection_exists(settings.qdrant_collection):
        logger.warning("Qdrant collection not found — creating it", collection=settings.qdrant_collection)
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Return a cached vector store singleton."""
    backend = settings.vector_store_backend
    logger.info("Initialising vector store", backend=backend)
    if backend == "faiss":
        return _build_faiss_store()
    if backend == "qdrant":
        return _build_qdrant_store()
    raise ValueError(f"Unsupported vector store backend: {backend!r}. Choose 'faiss' or 'qdrant'.")
