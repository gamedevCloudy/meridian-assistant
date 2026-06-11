"""
eval.run_judge

LLM-as-judge for answer correctness. Reuses the OpenRouter model the
agent already uses. For each case where the agent returned a final
answer (not a handoff), the judge rates:

  - grounded    0-1: does the answer rely on the retrieved KB content?
  - correct     0-1: does it answer the question factually?
  - cited       0-1: does it cite a source by name?

Run after run_action.py; reads results/actions.json.

    python -m eval.run_judge
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from app.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
ACTIONS_PATH = EVAL_DIR / "results" / "actions.json"
RESULTS_PATH = EVAL_DIR / "results" / "judge.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", Config.DEFAULT_LLM_MED)

# Targets — match eval/README.md
TARGET_GROUNDED = 0.80
TARGET_CORRECT = 0.80
TARGET_CITED = 0.80

JUDGE_PROMPT = """You are grading a customer-support assistant's reply. \
Reply with ONLY a JSON object — no prose, no markdown fences.

Score three dimensions on a 0-1 scale (two decimals):

  grounded:   Does the answer rely on the provided context, or did it invent
              facts not in the context? (1 = fully grounded, 0 = hallucinated)
  correct:    Does the answer actually address the customer's question correctly?
              (1 = correct and complete, 0 = wrong or misses the point)
  cited:      Does the answer cite at least one source by name (e.g.
              "Plumbing Pricing, p.1")? (1 = yes, 0 = no)

If the assistant produced a handoff (no answer to grade), return:
  {"skipped": true, "reason": "handoff"}

Context (retrieved KB snippets):
{context}

Customer question:
{question}

Assistant answer:
{answer}

JSON:"""


def build_judge() -> ChatOpenRouter:
    return ChatOpenRouter(model=JUDGE_MODEL, temperature=0, max_retries=3)


def grade(case_row: dict, judge: ChatOpenRouter) -> dict:
    answer = case_row.get("answer_excerpt") or ""
    if not answer or case_row.get("error"):
        return {"id": case_row["id"], "skipped": True, "reason": "no_answer"}
    if case_row.get("handoff_observed") and not answer.strip():
        return {"id": case_row["id"], "skipped": True, "reason": "handoff_only"}

    context = "(context unavailable — retrieval ran in a separate process)"
    msg = judge.invoke(
        [
            SystemMessage(content="You are a strict JSON-only grader."),
            HumanMessage(content=JUDGE_PROMPT.format(context=context, question=case_row.get("user_message", ""), answer=answer)),
        ]
    )
    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start : end + 1])
    except Exception:
        log.warning("Judge returned non-JSON for case %d: %s", case_row["id"], raw[:200])
        return {"id": case_row["id"], "skipped": True, "reason": "judge_parse_error", "raw": raw[:200]}

    return {
        "id": case_row["id"],
        "category": case_row["category"],
        "grounded": float(data.get("grounded", 0)),
        "correct": float(data.get("correct", 0)),
        "cited": float(data.get("cited", 0)),
    }


def aggregate(per_case: list[dict]) -> dict:
    scored = [c for c in per_case if not c.get("skipped")]
    if not scored:
        return {"cases_evaluated": 0}
    return {
        "cases_evaluated": len(scored),
        "grounded_mean": round(statistics.fmean(c["grounded"] for c in scored), 4),
        "correct_mean": round(statistics.fmean(c["correct"] for c in scored), 4),
        "cited_mean": round(statistics.fmean(c["cited"] for c in scored), 4),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    if not ACTIONS_PATH.exists():
        log.error("Missing %s — run eval.run_action first", ACTIONS_PATH)
        return 1
    actions = json.loads(ACTIONS_PATH.read_text())
    cases = actions["per_case"]

    test_cases = {c["id"]: c for c in json.loads((EVAL_DIR / "test_set.json").read_text())["cases"]}
    for row in cases:
        row["user_message"] = test_cases[row["id"]]["user_message"]

    log.info("Judging %d cases with model=%s", len(cases), JUDGE_MODEL)
    judge = build_judge()
    per_case = [grade(c, judge) for c in cases]
    summary = aggregate(per_case)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))
    log.info("Wrote %s", RESULTS_PATH)

    print("\n=== Judge metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.no_gate or not summary["cases_evaluated"]:
        return 0
    failures = []
    for k, target in (("grounded_mean", TARGET_GROUNDED), ("correct_mean", TARGET_CORRECT), ("cited_mean", TARGET_CITED)):
        if summary[k] < target:
            failures.append(f"{k}={summary[k]} < {target}")
    if failures:
        log.error("GATE FAILED: %s", "; ".join(failures))
        return 1
    log.info("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
