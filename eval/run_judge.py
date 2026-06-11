"""
eval.run_judge

LLM-as-judge for answer correctness. Reuses the OpenRouter model the
agent already uses. For each case where the agent returned a final
answer (not a handoff), the judge rates:

  - grounded    0-1: does the answer rely on the retrieved KB content?
  - correct     0-1: does it answer the question factually?
  - cited       0-1: does it cite a source by name?

Run after run_action.py; reads results/actions.json.
Uses asyncio with a semaphore for parallel execution.

    python -m eval.run_judge
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from app.config import Config
from app.data_loader.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
ACTIONS_PATH = EVAL_DIR / "results" / "actions.json"
RESULTS_PATH = EVAL_DIR / "results" / "judge.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")

# Concurrency limit for parallel judge calls
SEM_COUNT = int(os.environ.get("EVAL_JUDGE_SEM_COUNT", "10"))

# Targets — match eval/README.md
TARGET_GROUNDED = 0.80
TARGET_CORRECT = 0.80
TARGET_CITED = 0.80

JUDGE_PROMPT = """You are a strict but fair grader evaluating a customer-support assistant's reply.

Reply with ONLY a JSON object — no prose, no markdown fences. The JSON must contain these keys:
  "grounded" (0-1, two decimals)
  "correct" (0-1, two decimals)
  "cited" (0-1, two decimals)
  "explanation" (brief string, 1-2 sentences)

--- SCORING RUBRIC ---

GROUNDED (does the answer rely on the provided KB context?)
  1.0: Every factual claim is supported by the retrieved context.
  0.5: Mostly grounded, but contains one unsupported inference or minor extrapolation.
  0.0: Contains hallucinated facts not in the context, or contradicts the context.

CORRECT (does the answer address the customer's question factually?)
  1.0: Directly answers the question, correct and complete.
  0.5: Partially correct, or asks a clarifying question that moves toward the answer.
  0.0: Wrong answer, or completely misses the point.

CITED (does it cite sources by name?)
  1.0: Contains at least one citation in the format [Document Name, p.N] or [Document Name | p.N].
         OR: the response contains only clarifying questions with no factual claims (nothing to cite).
  0.5: Mentions document names inline but not in the exact bracket format.
  0.0: Contains factual claims but no citation at all.

--- FEW-SHOT EXAMPLES ---

Example 1 (score: grounded=1.0, correct=1.0, cited=1.0):
  Question: "What are your Saturday hours?"
  Answer: "The Herndon branch opens at 8:00 AM on Saturdays and closes at 2:00 PM. [Branch Hours, p.0]"
  → All facts from KB, directly answers, cites properly.

Example 2 (score: grounded=1.0, correct=0.5, cited=1.0):
  Question: "Can I book a panel inspection in Rockville?"
  Answer: "Rockville is in our service area. To book, I need your preferred date and time window. [Service Area North, p.0]"
  → Grounded and cited, but partially correct (asks for confirmation rather than giving direct answer).

Example 3 (score: grounded=0.0, correct=0.0, cited=0.0):
  Question: "My booking is tomorrow, what's the status?"
  Answer: "I've escalated your request to a human agent."
  → No factual content from KB, doesn't answer the question, no citation.

Example 4 (score: grounded=1.0, correct=1.0, cited=1.0):
  Question: "I need an electrician to look at a faulty outlet."
  Answer: "I'd be happy to help. To schedule this, could you provide your ZIP code, preferred date, and contact details?"
  → The response only asks clarifying questions with no factual claims, so citation is not required. Score cited=1.0.

--- CONTEXT FOR THIS CASE ---

Expected action type: {expected_action}
  - "answer" = direct answer expected
  - "answer_or_confirm" = answer OR ask for confirmation is acceptable
  - "answer_or_handoff" = answer OR handoff is acceptable
  - "handoff" = human handoff expected (skip grading)
  - "create_booking" = booking should be created

Retrieved KB context:
{context}

Customer question:
{question}

Assistant answer:
{answer}

JSON:"""


def build_judge() -> ChatOpenRouter:
    return ChatOpenRouter(model=JUDGE_MODEL, temperature=0, max_retries=3)


def grade_sync(case_row: dict, judge: ChatOpenRouter) -> dict:
    answer = case_row.get("answer_full") or case_row.get("answer_excerpt") or ""
    if not answer or case_row.get("error"):
        return {"id": case_row["id"], "skipped": True, "reason": "no_answer"}
    
    # Skip handoff cases — action eval already covers these via handoff_f1
    if case_row.get("handoff_observed"):
        return {"id": case_row["id"], "skipped": True, "reason": "handoff"}
    expected_action = case_row.get("expected_action", "")
    if expected_action == "handoff":
        return {"id": case_row["id"], "skipped": True, "reason": "expected_handoff"}

    # Retrieve context for this case so the judge can verify grounding
    question = case_row.get("user_message", "")
    try:
        docs = retrieve(question, k=4)
        context_chunks = [
            f"[{d.metadata.get('doc_name', 'unknown')} | p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        ]
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant documents retrieved)"
    except Exception as e:
        log.warning("Retrieval failed for case %d: %s", case_row["id"], e)
        context = "(retrieval failed)"

    msg = judge.invoke(
        [
            SystemMessage(content="You are a strict JSON-only grader."),
            HumanMessage(content=JUDGE_PROMPT.format(
                context=context, 
                question=question, 
                answer=answer,
                expected_action=expected_action
            )),
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
        "explanation": str(data.get("explanation", "")),
    }


async def grade(case_row: dict, judge: ChatOpenRouter, sem: asyncio.Semaphore) -> dict:
    async with sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, grade_sync, case_row, judge)


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


async def run_all(cases: list[dict], judge: ChatOpenRouter) -> list[dict]:
    sem = asyncio.Semaphore(SEM_COUNT)
    tasks = [grade(c, judge, sem) for c in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_case: list[dict] = []
    for case, result in zip(cases, results):
        if isinstance(result, Exception):
            log.error("Judge error on case %d: %s", case["id"], result)
            per_case.append({"id": case["id"], "skipped": True, "reason": "judge_exception", "error": str(result)})
        else:
            per_case.append(result)
    return per_case


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

    log.info("Judging %d cases with model=%s (sem=%s)", len(cases), JUDGE_MODEL, SEM_COUNT)
    judge = build_judge()
    per_case = asyncio.run(run_all(cases, judge))
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
