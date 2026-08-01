"""
Splunk HEC Log Handler
-----------------------
Ships structured JSON logs to Splunk via the HTTP Event Collector (HEC).
Configured as a standard Python logging handler — zero code changes
needed in agents or API routers.

Setup in .env:
    SPLUNK_HEC_URL=https://your-splunk-instance:8088
    SPLUNK_HEC_TOKEN=your-hec-token

The handler is automatically attached to the root structlog pipeline
when SPLUNK_HEC_URL is set. If not configured, the handler is a no-op.

Usage (automatic — no manual calls needed):
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("agent_step", incident_id="INC-001", risk_score=0.82)
    # → automatically shipped to Splunk if configured
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class SplunkHECHandler(logging.Handler):
    """
    Async Splunk HEC handler. Batches log events and ships them
    in a background thread to avoid blocking the main request path.
    """

    def __init__(
        self,
        hec_url: str,
        token: str,
        source: str = "incident-remediation-ai",
        sourcetype: str = "_json",
        index: str = "main",
        batch_size: int = 50,
        flush_interval: float = 5.0,
    ):
        super().__init__()
        self.hec_url    = hec_url.rstrip("/") + "/services/collector/event"
        self.token      = token
        self.source     = source
        self.sourcetype = sourcetype
        self.index      = index
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: Queue[dict] = Queue(maxsize=10_000)
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="splunk-hec")
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                payload = json.loads(msg)
            except (json.JSONDecodeError, TypeError):
                payload = {"message": str(msg)}

            event = {
                "time": datetime.now(timezone.utc).timestamp(),
                "host": "incident-ai",
                "source": self.source,
                "sourcetype": self.sourcetype,
                "index": self.index,
                "event": {
                    **payload,
                    "level": record.levelname,
                    "logger": record.name,
                },
            }
            self._queue.put_nowait(event)
        except Exception:
            self.handleError(record)

    def _flush_loop(self) -> None:
        import time
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            self._send_batch()

    def _send_batch(self) -> None:
        events: list[dict] = []
        try:
            while len(events) < self.batch_size:
                events.append(self._queue.get_nowait())
        except Empty:
            pass

        if not events:
            return

        try:
            import urllib.request
            body = "\n".join(json.dumps(e) for e in events).encode("utf-8")
            req  = urllib.request.Request(
                self.hec_url,
                data=body,
                headers={
                    "Authorization": f"Splunk {self.token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Splunk HEC non-200", status=resp.status)
        except Exception as exc:
            logger.warning("Splunk HEC send failed", error=str(exc), dropped=len(events))

    def close(self) -> None:
        self._stop.set()
        self._send_batch()   # Final flush
        super().close()


def setup_splunk_handler() -> bool:
    """
    Attach Splunk HEC handler to the root Python logger if configured.
    Returns True if handler was installed, False if SPLUNK_HEC_URL not set.
    """
    hec_url   = getattr(settings, "splunk_hec_url", None)
    hec_token = getattr(settings, "splunk_hec_token", None)

    if not hec_url:
        return False

    handler = SplunkHECHandler(
        hec_url=hec_url,
        token=hec_token or "",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("Splunk HEC handler installed", url=hec_url)
    return True


# Auto-install on import if configured
try:
    setup_splunk_handler()
except Exception as _exc:
    pass
