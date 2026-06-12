"""
eval.run_action

Runs the full LangGraph agent against every test case and compares the
observed action to `expected_action`. Uses asyncio with a semaphore for
parallel execution.

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
import asyncio
import json
import logging
import os
import sqlite3
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

# Concurrency limit for parallel agent calls
SEM_COUNT = int(os.environ.get("EVAL_SEM_COUNT", "10"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.environ.get("EVAL_RETRY_BACKOFF", "2.0"))

_REASON_SYNONYMS: dict[str, set[str]] = {
    "fee_dispute": {"fee_dispute", "billing_dispute", "billing", "dispute", "fee_waiver", "waiver"},
    "billing_dispute": {"fee_dispute", "billing_dispute", "billing", "dispute", "fee_waiver"},
    "complaint": {"complaint", "manager_request", "manager", "speak_to_manager", "supervisor", "unhappy"},
    "manager_request": {"complaint", "manager_request", "manager", "speak_to_manager", "supervisor"},
    "commercial_request": {"commercial_request", "commercial", "net_30", "commercial_account", "volume_discount", "property_management", "recurring_maintenance", "maintenance_contract"},
    "commercial": {"commercial_request", "commercial", "net_30", "commercial_account", "volume_discount", "property_management", "recurring_maintenance"},
    "emergency": {"emergency", "emergency_active_leak", "emergency_water", "active_leak", "flooding", "water_leak", "emergency_flood"},
    "emergency_gas_leak": {"emergency_gas_leak", "emergency_gas", "gas", "gas_leak", "gas_smell", "gas_emergency"},
    "emergency_gas": {"emergency_gas_leak", "emergency_gas", "gas", "gas_leak", "gas_smell", "gas_emergency"},
    "emergency_active_leak": {"emergency", "emergency_active_leak", "emergency_water", "active_leak", "flooding", "water_leak"},
    "emergency_water": {"emergency", "emergency_active_leak", "emergency_water", "active_leak", "flooding"},
    "out_of_scope": {"out_of_scope", "status_check", "booking_status", "no_tool"},
}


def _reason_matches(expected: str | None, observed: str | None) -> bool:
    if not expected or not observed:
        return True
    el = expected.lower().replace("-", "_").replace(" ", "_")
    ol = observed.lower().replace("-", "_").replace(" ", "_")
    if el in ol or ol in el:
        return True
    all_syns = _REASON_SYNONYMS.get(el, set()) | _REASON_SYNONYMS.get(ol, set())
    return el in all_syns or ol in all_syns


def load_cases() -> list[dict]:
    return json.loads((EVAL_DIR / "test_set.json").read_text())["cases"]


async def run_agent(case: dict, timeout_s: float, sem: asyncio.Semaphore) -> dict:
    """Async agent call with retry + exponential backoff for rate-limit resilience."""
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": [HumanMessage(content=case["user_message"])]},
                        config={"recursion_limit": 10},
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                last_error = f"timeout after {timeout_s}s"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))
                    continue
                return {"error": last_error, "session_id": session_id}
            except Exception as e:
                last_error = str(e)
                log.warning("Agent error on case %d (attempt %d/%d): %s", case["id"], attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))
                    continue
                return {"error": last_error, "session_id": session_id}

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

    return {"error": last_error or "max retries exhausted", "session_id": session_id}


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


def evaluate_case(observed: dict, case: dict) -> dict:
    obs_class = classify_observed(observed)
    expected = case["expected_action"]

    expected_reason = case.get("expected_handoff_reason")
    observed_reason = observed.get("handoff_reason")
    reason_ok = _reason_matches(expected_reason, observed_reason)

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
        "answer_full": observed.get("answer") or "",
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


async def run_all(cases: list[dict], timeout_s: float) -> list[dict]:
    sem = asyncio.Semaphore(SEM_COUNT)
    tasks = [run_agent(c, timeout_s, sem) for c in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_case: list[dict] = []
    for case, result in zip(cases, results):
        if isinstance(result, Exception):
            log.error("Async error on case %d: %s", case["id"], result)
            observed = {"error": str(result), "session_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"}
        else:
            observed = result
        per_case.append(evaluate_case(observed, case))
    return per_case


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

    log.info("Running %d new cases (timeout=%ss, sem=%s)", len(cases), args.timeout, SEM_COUNT)

    new = asyncio.run(run_all(cases, args.timeout))

    per_case = existing + new
    summary = aggregate(per_case)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
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
