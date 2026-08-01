"""
Ingestion Pipeline
------------------
Loads incident runbooks / historical incidents from S3 or local disk,
chunks them, embeds via Titan, and upserts into the vector store.

Run directly:
    python -m rag.ingestion.pipeline --source s3://my-bucket/runbooks/ --backend faiss

Or called from Airflow DAG: pipelines/airflow_dags/ingestion_dag.py
"""
from __future__ import annotations

# argparse Python's built-in standard library module used to create professional, user-friendly Command-Line Interfaces (CLIs)
import argparse
import os
from pathlib import Path
# Boto3 is a SDK for AWS services and is only required for S3 ingestion here, so we import it lazily to avoid unnecessary dependencies.
import boto3
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config.offline_mode  # noqa: F401 — must patch boto3/embeddings before rag imports below
from config.settings import get_settings
from rag.embeddings.bedrock import get_embeddings
from rag.vector_store.factory import get_vector_store

logger = structlog.get_logger(__name__)
settings = get_settings()

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_from_local(source_dir: str) -> list[Document]:
    docs: list[Document] = []
    for fp in Path(source_dir).rglob("*.txt"):
        text = fp.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(fp), "type": "local"}))
    logger.info("Loaded local documents", count=len(docs), path=source_dir)
    return docs


def _load_from_s3(s3_uri: str) -> list[Document]:
    """Load .txt files from an S3 prefix."""
    s3 = boto3.client("s3", region_name=settings.aws_region)
    # Parse s3://bucket/prefix
    uri = s3_uri.removeprefix("s3://")
    bucket, _, prefix = uri.partition("/")

    paginator = s3.get_paginator("list_objects_v2")
    docs: list[Document] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".txt"):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            docs.append(Document(page_content=body, metadata={"source": f"s3://{bucket}/{key}", "type": "s3"}))

    logger.info("Loaded S3 documents", count=len(docs), uri=s3_uri)
    return docs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(source: str, dry_run: bool = False) -> int:
    """
    Ingest documents from `source` into the configured vector store.

    Args:
        source: Local directory path or s3://bucket/prefix.
        dry_run: If True, skip upsert (useful for validation).

    Returns:
        Number of chunks indexed.
    """
    logger.info("Starting ingestion", source=source, backend=settings.vector_store_backend)

    # Load
    raw_docs = _load_from_s3(source) if source.startswith("s3://") else _load_from_local(source)
    if not raw_docs:
        logger.warning("No documents found", source=source)
        return 0

    # Chunk
    chunks = _SPLITTER.split_documents(raw_docs)
    logger.info("Documents chunked", raw=len(raw_docs), chunks=len(chunks))

    if dry_run:
        logger.info("Dry run — skipping upsert")
        return len(chunks)

    # Upsert
    store = get_vector_store()
    store.add_documents(chunks)

    if settings.vector_store_backend == "faiss":
        store.save_local(settings.faiss_index_path)
        logger.info("FAISS index persisted", path=settings.faiss_index_path)

    logger.info("Ingestion complete", chunks_indexed=len(chunks))
    return len(chunks)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--source", required=True, help="Local dir or s3://bucket/prefix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = run_ingestion(args.source, dry_run=args.dry_run)
    print(f"Indexed {n} chunks.")
