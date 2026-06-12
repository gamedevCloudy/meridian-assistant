"""
eval.run_retrieval

Computes retrieval-only metrics over the test set: Hit@3, Hit@5, MRR, Recall@k.
No LLM involved.

Run:
    python -m eval.run_retrieval [--no-gate]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from app.data_loader.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "results" / "retrieval.json"
K_VALUES = (3, 5)

# Targets — match eval/README.md
TARGET_HIT_AT_3 = 0.85
TARGET_MRR = 0.75


def load_cases() -> list[dict]:
    return json.loads((EVAL_DIR / "test_set.json").read_text())["cases"]


def source_matches(retrieved_source: str, expected: list[str]) -> bool:
    if not expected:
        return True
    rs = retrieved_source.lower().replace("\\", "/")
    for e in expected:
        e_lower = e.lower().replace("_", " ").replace(".pdf", "").strip()
        if e_lower in rs or rs.endswith(e_lower):
            return True
        stem = Path(retrieved_source).stem.lower().replace("_", " ")
        if e_lower in stem or stem in e_lower:
            return True
        parent = Path(retrieved_source).parent.name.lower()
        if e_lower in parent or parent in e_lower:
            return True
    return False


def first_relevant_rank(sources: list[str], expected: list[str]) -> int | None:
    for i, s in enumerate(sources, start=1):
        if source_matches(s, expected):
            return i
    return None


def evaluate_case(case: dict) -> dict:
    expected = case.get("expected_sources") or []
    if not expected:
        return {"id": case["id"], "skipped": True, "reason": "no expected_sources"}

    docs = retrieve(case["user_message"], k=max(K_VALUES))
    sources = [d.metadata.get("source", "") for d in docs]
    rank = first_relevant_rank(sources, expected)

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_sources": expected,
        "retrieved_sources": sources[: max(K_VALUES)],
        "first_relevant_rank": rank,
        **{f"hit_at_{k}": rank is not None and rank <= k for k in K_VALUES},
        **{f"recall_at_{k}": 1.0 if (rank is not None and rank <= k) else 0.0 for k in K_VALUES},
    }


def aggregate(per_case: list[dict]) -> dict:
    scored = [c for c in per_case if not c.get("skipped")]
    out: dict[str, int | float | None] = {
        "cases_evaluated": len(scored),
        "cases_total": len(per_case),
    }
    for k in K_VALUES:
        hits = [c[f"hit_at_{k}"] for c in scored]
        out[f"hit_at_{k}"] = round(sum(hits) / len(hits), 4) if hits else None
        out[f"recall_at_{k}"] = round(sum(c[f"recall_at_{k}"] for c in scored) / len(scored), 4) if scored else None
    mrr_vals = [1.0 / c["first_relevant_rank"] for c in scored if c["first_relevant_rank"]]
    out["mrr"] = round(statistics.fmean(mrr_vals), 4) if mrr_vals else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    log.info("Evaluating retrieval on %d cases", len(cases))
    per_case = [evaluate_case(c) for c in cases]
    summary = aggregate(per_case)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))
    log.info("Wrote %s", RESULTS_PATH)

    print("\n=== Retrieval metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.no_gate or not summary["cases_evaluated"]:
        return 0
    failures = []
    if summary["hit_at_3"] < TARGET_HIT_AT_3:
        failures.append(f"hit_at_3={summary['hit_at_3']} < {TARGET_HIT_AT_3}")
    if summary["mrr"] < TARGET_MRR:
        failures.append(f"mrr={summary['mrr']} < {TARGET_MRR}")
    if failures:
        log.error("GATE FAILED: %s", "; ".join(failures))
        return 1
    log.info("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
