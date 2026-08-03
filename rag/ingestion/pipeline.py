"""
Ingestion Pipeline
------------------
Loads incident runbooks / historical incidents from S3 or local disk,
chunks them, embeds via Titan, and upserts into the vector store.

Chunking is hierarchical and document-aware:
  1. Each document is first split into sections along its own structure —
     markdown headers (#, ##, ...) or the ALL-CAPS section-header style used
     by the sample runbooks (SYMPTOMS, ROOT CAUSES, REMEDIATION STEPS, ...).
     A document with no recognisable headers becomes one section, so
     unstructured docs still ingest fine.
  2. Each section is the "parent" unit. If a section is small enough to embed
     as-is, it has exactly one child chunk (itself). If it's larger than
     _CHUNK_SIZE, it's recursively split (same RecursiveCharacterTextSplitter
     as before) into smaller "child" chunks — these are what actually get
     embedded and searched, for precision.
  3. Every child chunk carries its full parent section in metadata
     (parent_id, parent_content, section_title), so the retriever can match
     on a precise fragment but return the whole section as context — see
     agents/retriever/agent.py.

Run directly:
    python -m rag.ingestion.pipeline --source s3://my-bucket/runbooks/ --backend faiss

Or called from Airflow DAG: pipelines/airflow_dags/ingestion_dag.py
"""
from __future__ import annotations

# argparse Python's built-in standard library module used to create professional, user-friendly Command-Line Interfaces (CLIs)
import argparse
import re
from pathlib import Path
# Boto3 is a SDK for AWS services and is only required for S3 ingestion here, so we import it lazily to avoid unnecessary dependencies.
import boto3
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config.offline_mode  # noqa: F401 — must patch boto3/embeddings before rag imports below
from config.settings import get_settings
from rag.vector_store.factory import get_vector_store

logger = structlog.get_logger(__name__)
settings = get_settings()

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
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


# ── Hierarchical, document-aware chunking ────────────────────────────────────

_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$")
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")
# A line is a section header if, once any trailing "(clarification)" is
# stripped, what's left is short and entirely uppercase letters/spaces/
# hyphens/slashes — e.g. "SYMPTOMS", "ROOT CAUSES (most common)",
# "POST-INCIDENT". This deliberately excludes label-like lines such as
# "ESCALATION: DBA team if DB is unresponsive after step 3." (has lowercase
# content) so real sentences aren't mistaken for headers.
_CAPS_HEADER_CORE_RE = re.compile(r"[A-Z][A-Z /\-]*")


def _looks_like_header(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    md = _MARKDOWN_HEADER_RE.match(stripped)
    if md:
        return md.group(1).strip()

    core = _PARENTHETICAL_RE.sub("", stripped)
    if core and not core.endswith(":") and _CAPS_HEADER_CORE_RE.fullmatch(core):
        return stripped

    return None


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split raw document text into (section_title, section_text) pairs.

    Text before the first recognised header is kept as a "Preamble" section
    rather than dropped. A document with no headers at all yields a single
    Preamble section covering the whole text — chunking then falls back to
    the same flat recursive split used before this change.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        header = _looks_like_header(line)
        if header:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [
        (title, "\n".join(lines).strip())
        for title, lines in sections
        if "\n".join(lines).strip()
    ]


def _chunk_document_hierarchically(doc: Document) -> list[Document]:
    """Split one loaded Document into child chunks for embedding, each
    carrying its full parent section in metadata (parent_id, parent_content,
    section_title) so the retriever can return the whole section instead of
    the isolated fragment it matched on.
    """
    source = doc.metadata.get("source", "unknown")
    children: list[Document] = []

    for i, (title, section_text) in enumerate(_split_into_sections(doc.page_content)):
        parent_id = f"{source}::section-{i}"
        sub_texts = (
            _SPLITTER.split_text(section_text)
            if len(section_text) > _CHUNK_SIZE
            else [section_text]
        )
        for j, sub_text in enumerate(sub_texts):
            children.append(Document(
                page_content=sub_text,
                metadata={
                    **doc.metadata,
                    "section_title": title,
                    "parent_id": parent_id,
                    "parent_content": section_text,
                    "chunk_index": j,
                },
            ))

    return children


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

    # Chunk — hierarchical + document-aware (see module docstring)
    chunks: list[Document] = []
    for doc in raw_docs:
        chunks.extend(_chunk_document_hierarchically(doc))
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
