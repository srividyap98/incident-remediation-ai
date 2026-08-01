"""
Prompt Regression Testing Framework
-------------------------------------
Runs a fixed set of benchmark incidents through the live workflow
and compares outputs against golden answers using semantic similarity.

Usage:
    python -m evaluation.prompt_regression
    python -m evaluation.prompt_regression --threshold 0.75
    python -m evaluation.prompt_regression --save-baseline

CI integration:
    Exit code 0 = pass, Exit code 1 = regression detected.
    Add to CI: python -m evaluation.prompt_regression || exit 1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_BASELINE_PATH = Path("data/regression_baseline.json")
_DEFAULT_THRESHOLD = 0.75   # Min semantic similarity to pass

# ── Golden benchmark incidents ────────────────────────────────────────────────
BENCHMARKS: list[dict[str, Any]] = [
    {
        "input": {
            "incident_id": "BENCH-001",
            "incident_title": "PostgreSQL connection pool exhausted",
            "incident_description": "All DB connections refused. Service returning 503.",
            "severity": "P1",
            "affected_service": "payments-api",
            "raw_logs": ["FATAL: sorry, too many clients already", "ERROR: connection pool exhausted"],
        },
        "golden": {
            "root_cause_keywords": ["connection pool", "connections", "database", "exhausted", "leak"],
            "remediation_keywords": ["restart", "connections", "pool size", "pg_terminate_backend"],
            "min_steps": 3,
            "max_risk_score": 1.0,
            "min_risk_score": 0.5,
        },
    },
    {
        "input": {
            "incident_id": "BENCH-002",
            "incident_title": "Kubernetes pod OOMKilled",
            "incident_description": "order-service pods killed every 2 hours with OOMKilled reason.",
            "severity": "P2",
            "affected_service": "order-service",
            "raw_logs": ["OOMKilled: container exceeded memory limit 3Gi"],
        },
        "golden": {
            "root_cause_keywords": ["memory", "leak", "OOM", "limit", "heap"],
            "remediation_keywords": ["memory", "rollback", "limit", "restart", "kubectl"],
            "min_steps": 2,
            "max_risk_score": 0.9,
            "min_risk_score": 0.3,
        },
    },
    {
        "input": {
            "incident_id": "BENCH-003",
            "incident_title": "Kafka consumer lag 800k messages",
            "incident_description": "inventory-updater consumer group 4 hours behind on order-events topic.",
            "severity": "P2",
            "affected_service": "inventory-service",
            "raw_logs": ["ERROR: Failed to deserialize message at offset 2847201"],
        },
        "golden": {
            "root_cause_keywords": ["poison", "message", "offset", "deserializ", "consumer"],
            "remediation_keywords": ["offset", "skip", "DLQ", "dead letter", "consumer"],
            "min_steps": 3,
            "max_risk_score": 0.9,
            "min_risk_score": 0.3,
        },
    },
]


@dataclass
class BenchmarkResult:
    incident_id: str
    passed: bool
    rca_keyword_hits: int
    rca_keyword_total: int
    remediation_keyword_hits: int
    remediation_keyword_total: int
    steps_count: int
    risk_score: float
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        kw_total = self.rca_keyword_total + self.remediation_keyword_total
        kw_hits = self.rca_keyword_hits + self.remediation_keyword_hits
        return kw_hits / kw_total if kw_total > 0 else 0.0


def _keyword_hit_count(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _evaluate_result(benchmark: dict[str, Any], result: Mapping[str, Any]) -> BenchmarkResult:
    golden = benchmark["golden"]
    incident_id = benchmark["input"]["incident_id"]
    failures: list[str] = []

    rca = result.get("root_cause_analysis", "") or ""
    steps = result.get("remediation_steps", [])
    steps_text = " ".join(steps)
    risk = result.get("risk_score") or 0.0

    # Keyword checks
    rca_hits = _keyword_hit_count(rca, golden["root_cause_keywords"])
    rem_hits = _keyword_hit_count(steps_text, golden["remediation_keywords"])
    rca_total = len(golden["root_cause_keywords"])
    rem_total = len(golden["remediation_keywords"])

    # Validate
    if len(steps) < golden["min_steps"]:
        failures.append(f"Too few remediation steps: got {len(steps)}, expected >= {golden['min_steps']}")

    if not (golden["min_risk_score"] <= risk <= golden["max_risk_score"]):
        failures.append(f"Risk score {risk:.2f} outside expected range [{golden['min_risk_score']}, {golden['max_risk_score']}]")

    if rca_hits < rca_total * 0.4:
        failures.append(f"RCA keyword hits low: {rca_hits}/{rca_total}")

    if rem_hits < rem_total * 0.4:
        failures.append(f"Remediation keyword hits low: {rem_hits}/{rem_total}")

    if result.get("status") == "error":
        failures.append(f"Workflow error: {result.get('error')}")

    return BenchmarkResult(
        incident_id=incident_id,
        passed=len(failures) == 0,
        rca_keyword_hits=rca_hits,
        rca_keyword_total=rca_total,
        remediation_keyword_hits=rem_hits,
        remediation_keyword_total=rem_total,
        steps_count=len(steps),
        risk_score=risk,
        failures=failures,
    )


def run_regression(threshold: float = _DEFAULT_THRESHOLD, save_baseline: bool = False) -> bool:
    """
    Run all benchmark incidents and return True if all pass.

    Args:
        threshold: Minimum average keyword score to pass (0.0–1.0).
        save_baseline: If True, save current results as the new baseline.

    Returns:
        True if regression tests pass, False if regression detected.
    """
    from agents.orchestrator.graph import run_incident_workflow
    import mlflow
    from config.settings import get_settings
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    results: list[BenchmarkResult] = []
    print(f"\nRunning {len(BENCHMARKS)} regression benchmarks...")
    print("=" * 60)

    with mlflow.start_run(run_name=f"regression-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"):
        for i, benchmark in enumerate(BENCHMARKS, 1):
            iid = benchmark["input"]["incident_id"]
            print(f"\n[{i}/{len(BENCHMARKS)}] {iid} — {benchmark['input']['incident_title']}")

            try:
                state = run_incident_workflow(benchmark["input"], thread_id=f"regression-{iid}")
                br = _evaluate_result(benchmark, state or {})
            except Exception as exc:
                br = BenchmarkResult(
                    incident_id=iid, passed=False,
                    rca_keyword_hits=0, rca_keyword_total=len(benchmark["golden"]["root_cause_keywords"]),
                    remediation_keyword_hits=0, remediation_keyword_total=len(benchmark["golden"]["remediation_keywords"]),
                    steps_count=0, risk_score=0.0,
                    failures=[f"Exception: {exc}"],
                )

            results.append(br)
            status = "✓ PASS" if br.passed else "✗ FAIL"
            print(f"  {status} | score={br.score:.2f} | steps={br.steps_count} | risk={br.risk_score:.2f}")
            if br.failures:
                for f in br.failures:
                    print(f"    → {f}")

            mlflow.log_metrics({
                f"{iid}_score": br.score,
                f"{iid}_passed": float(br.passed),
                f"{iid}_risk_score": br.risk_score,
            })

        avg_score = float(np.mean([r.score for r in results]))
        all_passed = all(r.passed for r in results)
        overall = avg_score >= threshold and all_passed

        mlflow.log_metrics({"avg_score": avg_score, "all_passed": float(all_passed), "overall_pass": float(overall)})
        mlflow.log_param("threshold", threshold)

    print("\n" + "=" * 60)
    print(f"Average score: {avg_score:.2f} (threshold: {threshold})")
    print(f"All passed:    {all_passed}")
    print(f"Overall:       {'PASS ✓' if overall else 'FAIL ✗ — REGRESSION DETECTED'}")
    print("=" * 60 + "\n")

    if save_baseline:
        baseline = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "avg_score": avg_score,
            "results": [
                {"incident_id": r.incident_id, "score": r.score, "passed": r.passed}
                for r in results
            ],
        }
        _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"Baseline saved to {_BASELINE_PATH}")

    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt regression tests")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    passed = run_regression(args.threshold, args.save_baseline)
    sys.exit(0 if passed else 1)
