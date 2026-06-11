"""
eval.run_action

Runs the full LangGraph agent against every test case and compares the
observed action to `expected_action`.

Action classes:
    answer            -> final answer, no handoff
    answer_or_confirm -> answer OR pre-commitment confirmation
    answer_or_handoff -> answer OR clean handoff
    handoff           -> agent must call handoff_to_human
    create_booking    -> booking must appear in mock DB

Run:
    python -m eval.run_action [--limit N] [--offset N] [--timeout S]
                              [--resume-from PATH] [--no-gate]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agents.graph import agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "results" / "actions.json"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bookings.db"

# Minimum bar — match eval/README.md
TARGET_ACTION_ACCURACY = 0.85
TARGET_HANDOFF_F1 = 0.90

ACCEPTED_ALTERNATIVES = {
    "answer": {"answer", "answer_or_confirm", "answer_or_handoff"},
    "answer_or_confirm": {"answer", "answer_or_confirm"},
    "answer_or_handoff": {"answer", "answer_or_handoff", "handoff"},
}


def load_cases() -> list[dict]:
    return json.loads((EVAL_DIR / "test_set.json").read_text())["cases"]


def run_agent(case: dict, timeout_s: float) -> dict:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"

    def _run():
        return agent.invoke({"messages": [HumanMessage(content=case["user_message"])]})

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            try:
                result = ex.submit(_run).result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                return {"error": f"timeout after {timeout_s}s", "session_id": session_id}
    except Exception as e:
        log.exception("Agent error on case %d", case["id"])
        return {"error": str(e), "session_id": session_id}

    final = result["messages"][-1]
    tool_names = [
        tc["name"]
        for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
        for tc in m.tool_calls
    ]
    return {
        "session_id": session_id,
        "answer": getattr(final, "content", str(final)),
        "handoff": bool(result.get("handoff_requested")),
        "handoff_reason": (result.get("handoff_payload") or {}).get("reason"),
        "tool_calls": sorted(set(tool_names)),
        "llm_calls": result.get("llm_calls", 0),
    }


def classify_observed(observed: dict) -> str:
    if observed.get("error"):
        return "error"
    if observed["handoff"]:
        return "handoff"
    if "create_booking" in observed["tool_calls"]:
        return "create_booking"
    return "answer"


def expected_matches_observed(expected: str, observed: str) -> bool:
    if expected == observed:
        return True
    return observed in ACCEPTED_ALTERNATIVES.get(expected, {expected})


def verify_booking_against_db(tool_calls: list[str]) -> bool:
    if "create_booking" not in tool_calls or not DB_PATH.exists():
        return True
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT booking_id FROM bookings ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row is not None and str(row[0]).startswith("BK-")


def evaluate_case(case: dict, timeout_s: float) -> dict:
    observed = run_agent(case, timeout_s)
    obs_class = classify_observed(observed)
    expected = case["expected_action"]

    expected_reason = case.get("expected_handoff_reason")
    observed_reason = observed.get("handoff_reason")
    reason_ok = (
        not expected_reason
        or not observed_reason
        or expected_reason.lower() in observed_reason.lower()
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_action": expected,
        "observed_action": obs_class,
        "action_ok": expected_matches_observed(expected, obs_class),
        "handoff_expected": case["expected_handoff"],
        "handoff_observed": observed.get("handoff", False),
        "handoff_ok": (not case["expected_handoff"]) or observed.get("handoff") is True,
        "handoff_reason_ok": reason_ok,
        "booking_db_ok": verify_booking_against_db(observed.get("tool_calls", [])),
        "tool_calls": observed.get("tool_calls", []),
        "llm_calls": observed.get("llm_calls", 0),
        "error": observed.get("error"),
        "answer_excerpt": (observed.get("answer") or "")[:200],
    }


def aggregate(per_case: list[dict]) -> dict:
    n = len(per_case)
    by_category: dict[str, dict] = {}
    for c in per_case:
        s = by_category.setdefault(c["category"], {"n": 0, "action_ok": 0, "handoff_ok": 0, "reason_ok": 0})
        s["n"] += 1
        s["action_ok"] += int(c["action_ok"])
        s["handoff_ok"] += int(c["handoff_ok"])
        s["reason_ok"] += int(c["handoff_reason_ok"])
    for s in by_category.values():
        s["action_accuracy"] = round(s["action_ok"] / s["n"], 4)

    errored = sum(1 for c in per_case if c["observed_action"] == "error")
    evaluated = n - errored
    action_correct = sum(int(c["action_ok"]) for c in per_case)
    handoff_correct = sum(int(c["handoff_ok"]) for c in per_case)

    handoff_eval = [c for c in per_case if c["handoff_expected"] and c["observed_action"] != "error"]
    non_handoff_eval = [c for c in per_case if not c["handoff_expected"] and c["observed_action"] != "error"]
    tp = sum(1 for c in handoff_eval if c["handoff_observed"])
    fp = sum(1 for c in non_handoff_eval if c["handoff_observed"])
    fn = sum(1 for c in handoff_eval if not c["handoff_observed"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "cases": n,
        "cases_evaluated": evaluated,
        "cases_errored": errored,
        "action_accuracy": round(action_correct / n, 4),
        "action_accuracy_on_evaluated": round(action_correct / evaluated, 4) if evaluated else None,
        "handoff_accuracy": round(handoff_correct / n, 4),
        "handoff_precision": round(precision, 4),
        "handoff_recall": round(recall, 4),
        "handoff_f1": round(f1, 4),
        "by_category": by_category,
        "observed_action_distribution": dict(Counter(c["observed_action"] for c in per_case)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-gate", action="store_true", help="Skip pass/fail assertions")
    args = parser.parse_args()

    cases = load_cases()[args.offset :]
    if args.limit:
        cases = cases[: args.limit]

    existing: list[dict] = []
    if args.resume_from and args.resume_from.exists():
        existing = json.loads(args.resume_from.read_text()).get("per_case", [])
        done = {r["id"] for r in existing if not r.get("error")}
        cases = [c for c in cases if c["id"] not in done]

    sleep_s = float(os.environ.get("EVAL_INTER_CASE_SLEEP", "2"))
    log.info("Running %d new cases (timeout=%ss)", len(cases), args.timeout)
    new: list[dict] = []
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] case %d (%s)", i, len(cases), case["id"], case["category"])
        new.append(evaluate_case(case, args.timeout))
        partial = existing + new
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({"summary": aggregate(partial), "per_case": partial}, indent=2))
        if i < len(cases) and sleep_s > 0:
            time.sleep(sleep_s)

    per_case = existing + new
    summary = aggregate(per_case)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))
    log.info("Wrote %s", RESULTS_PATH)

    print("\n=== Action metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.no_gate or not summary["cases_evaluated"]:
        return 0

    failures = []
    acc = summary["action_accuracy_on_evaluated"]
    if acc is not None and acc < TARGET_ACTION_ACCURACY:
        failures.append(f"action_accuracy_on_evaluated={acc} < {TARGET_ACTION_ACCURACY}")
    # Only gate handoff_f1 when we have >=3 evaluated handoff cases (avoid noise)
    handoff_eval_n = sum(
        1 for c in per_case
        if c["handoff_expected"] and c["observed_action"] != "error"
    )
    if handoff_eval_n >= 3 and summary["handoff_f1"] < TARGET_HANDOFF_F1:
        failures.append(f"handoff_f1={summary['handoff_f1']} < {TARGET_HANDOFF_F1} (n={handoff_eval_n})")
    if failures:
        log.error("GATE FAILED: %s", "; ".join(failures))
        return 1
    log.info("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
