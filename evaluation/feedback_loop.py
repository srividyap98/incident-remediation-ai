"""
Feedback Loop — Aggregation & Analysis
---------------------------------------
Reads feedback collected by api/routers/feedback.py and produces
improvement reports identifying patterns in low-rated responses.

Run:
    python -m evaluation.feedback_loop
    python -m evaluation.feedback_loop --min-ratings 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_FEEDBACK_STORE = Path("data/feedback.jsonl")
_REPORT_PATH    = Path("data/feedback_report.json")

_LOW_RATING_THRESHOLD = 3   # Ratings <= this are considered low quality


def load_feedback() -> list[dict]:
    if not _FEEDBACK_STORE.exists():
        return []
    records = []
    with _FEEDBACK_STORE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def aggregate(records: list[dict]) -> dict:
    """Compute summary statistics and identify low-quality patterns."""
    if not records:
        return {"total_feedback": 0, "message": "No feedback recorded yet."}

    # Overall stats
    rca_scores  = [r["rca_accuracy"]           for r in records if "rca_accuracy" in r]
    rem_scores  = [r["remediation_usefulness"] for r in records if "remediation_usefulness" in r]
    avg_rca     = sum(rca_scores) / len(rca_scores) if rca_scores else 0
    avg_rem     = sum(rem_scores) / len(rem_scores)  if rem_scores else 0

    # By incident ID
    by_incident: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_incident[r["incident_id"]].append(r.get("average_rating", 0))

    low_incidents = {
        iid: round(sum(scores) / len(scores), 2)
        for iid, scores in by_incident.items()
        if sum(scores) / len(scores) <= _LOW_RATING_THRESHOLD
    }

    # Corrections provided
    corrections = [
        {"incident_id": r["incident_id"], "notes": r["correction_notes"], "reviewer": r.get("reviewer_id")}
        for r in records
        if r.get("correction_notes")
    ]

    # Rating distribution
    distribution = {str(i): sum(1 for r in records if r.get("average_rating", 0) >= i - 0.5 and r.get("average_rating", 0) < i + 0.5) for i in range(1, 6)}

    summary = {
        "total_feedback": len(records),
        "average_rca_accuracy": round(avg_rca, 2),
        "average_remediation_usefulness": round(avg_rem, 2),
        "overall_average": round((avg_rca + avg_rem) / 2, 2),
        "rating_distribution": distribution,
        "low_rated_incidents": low_incidents,
        "corrections_provided": len(corrections),
        "correction_samples": corrections[:5],   # First 5 for report preview
        "recommendations": _generate_recommendations(avg_rca, avg_rem, low_incidents, corrections),
    }
    return summary


def _generate_recommendations(avg_rca: float, avg_rem: float, low_incidents: dict, corrections: list) -> list[str]:
    recs = []
    if avg_rca < 3.5:
        recs.append(f"Root cause accuracy average is {avg_rca:.1f}/5 — review the risk evaluator prompt and consider adding more diverse runbooks to the knowledge base.")
    if avg_rem < 3.5:
        recs.append(f"Remediation usefulness average is {avg_rem:.1f}/5 — consider adding step-by-step specificity guidelines to the response generator system prompt.")
    if len(low_incidents) > 0:
        recs.append(f"{len(low_incidents)} incidents have average ratings <= {_LOW_RATING_THRESHOLD}: {list(low_incidents.keys())[:3]}. Review these specific cases for prompt improvement opportunities.")
    if len(corrections) >= 3:
        recs.append(f"{len(corrections)} corrections have been submitted by engineers — these should be reviewed as candidates for new runbook entries.")
    if not recs:
        recs.append("Quality metrics look healthy. Continue monitoring.")
    return recs


def run_analysis(min_ratings: int = 1) -> dict:
    records = load_feedback()
    if len(records) < min_ratings:
        print(f"Not enough feedback yet ({len(records)} records, minimum {min_ratings}). Collect more feedback first.")
        return {}

    report = aggregate(records)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\nFeedback Analysis Report")
    print("=" * 50)
    print(f"Total feedback records:    {report['total_feedback']}")
    print(f"Avg RCA accuracy:          {report['average_rca_accuracy']}/5")
    print(f"Avg remediation quality:   {report['average_remediation_usefulness']}/5")
    print(f"Overall average:           {report['overall_average']}/5")
    print(f"Low-rated incidents:       {len(report.get('low_rated_incidents', {}))}")
    print(f"Corrections submitted:     {report['corrections_provided']}")
    print("\nRecommendations:")
    for rec in report.get("recommendations", []):
        print(f"  → {rec}")
    print(f"\nFull report saved to {_REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse collected feedback")
    parser.add_argument("--min-ratings", type=int, default=1)
    args = parser.parse_args()
    run_analysis(args.min_ratings)
