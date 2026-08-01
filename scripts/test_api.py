"""
Quick API Test Script
---------------------
Fires all 6 seed incidents at the local API and prints results.
Requires the API to be running: make dev

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --incident 0   # run just the first one
    python scripts/test_api.py --all          # run all 6 sequentially
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

API_BASE = "http://localhost:8000"
INCIDENTS_FILE = Path("data/seed_incidents.json")


def check_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        data = r.json()
        print(f"  API status: {data['status']} | model: {data['bedrock_model']}")
        print(f"  Vector store: {data['vector_store_backend']}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ERROR: API not reachable at {API_BASE} — {e}")
        print("  Start it with: make dev")
        return False


def submit_incident(incident: dict) -> dict:
    r = httpx.post(
        f"{API_BASE}/api/v1/incidents/",
        json=incident,
        timeout=120,  # LLM calls can take time
    )
    r.raise_for_status()
    return r.json()


def resume_hitl(thread_id: str, approved: bool = True) -> dict:
    r = httpx.post(
        f"{API_BASE}/api/v1/hitl/resume",
        json={"thread_id": thread_id, "approved": approved, "reviewer_id": "dev-test"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def print_result(incident: dict, result: dict) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  {incident['incident_id']} | {incident['severity']} | {incident['affected_service']}")
    print(f"  {incident['incident_title']}")
    print(sep)
    print(f"  Status:        {result.get('status', 'N/A')}")
    print(f"  Risk score:    {result.get('risk_score', 'N/A')}")
    print(f"  HITL:          {result.get('requires_human_review', False)}")
    print(f"  Confidence:    {result.get('confidence_score', 'N/A')}")

    steps = result.get("remediation_steps", [])
    if steps:
        print(f"\n  Remediation steps ({len(steps)}):")
        for i, s in enumerate(steps[:3], 1):
            print(f"    {i}. {s[:80]}{'...' if len(s) > 80 else ''}")
        if len(steps) > 3:
            print(f"    ... and {len(steps) - 3} more")

    rca = result.get("root_cause_analysis")
    if rca:
        print(f"\n  Root cause: {rca[:120]}{'...' if len(rca) > 120 else ''}")

    factors = result.get("risk_factors", [])
    if factors:
        print(f"\n  Risk factors: {', '.join(factors[:3])}")

    error = result.get("error")
    if error:
        print(f"\n  ERROR: {error}")


def run_incident(incident: dict, index: int, total: int) -> None:
    print(f"\n[{index}/{total}] Submitting: {incident['incident_id']} ({incident['severity']})")
    start = time.perf_counter()
    try:
        result = submit_incident(incident)
        elapsed = time.perf_counter() - start
        print(f"  Completed in {elapsed:.1f}s")
        print_result(incident, result)

        # If HITL triggered, auto-approve in dev mode
        if result.get("status") == "pending_review":
            print("\n  [HITL] High-risk incident paused for review.")
            print("  [HITL] Auto-approving in dev mode...")
            resumed = resume_hitl(result["thread_id"], approved=True)
            print_result(incident, resumed)

    except httpx.HTTPStatusError as e:
        print(f"  HTTP error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Incident Remediation AI API")
    parser.add_argument("--incident", type=int, default=None, help="Index of incident to run (0-5)")
    parser.add_argument("--all", action="store_true", help="Run all 6 seed incidents")
    args = parser.parse_args()

    print("\nIncident Remediation AI — API Test Runner")
    print("=" * 56)
    print("\nChecking API health...")
    if not check_health():
        sys.exit(1)

    if not INCIDENTS_FILE.exists():
        print(f"\nERROR: {INCIDENTS_FILE} not found.")
        print("Run first: python scripts/seed_dev_db.py")
        sys.exit(1)

    incidents = json.loads(INCIDENTS_FILE.read_text())

    if args.incident is not None:
        if args.incident >= len(incidents):
            print(f"ERROR: Only {len(incidents)} incidents available (0–{len(incidents)-1})")
            sys.exit(1)
        run_incident(incidents[args.incident], 1, 1)
    elif args.all:
        for i, incident in enumerate(incidents, 1):
            run_incident(incident, i, len(incidents))
            if i < len(incidents):
                print("\n  Waiting 2s before next incident...")
                time.sleep(2)
    else:
        # Default: run just the first one
        print("\nNo flag given — running first incident. Use --all to run all 6.\n")
        run_incident(incidents[0], 1, 1)

    print("\n\nDone. Check MLflow for experiment results: make mlflow-ui")


if __name__ == "__main__":
    main()
