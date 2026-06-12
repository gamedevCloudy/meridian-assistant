"""
eval.run_eval

Thin orchestrator. Runs retrieval and action stages in parallel (they are
independent), then the judge stage (depends on action output), and stitches
their JSON outputs into a single summary.md.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --skip-judge
    python -m eval.run_eval --no-gate
    python -m eval.run_eval --action-args "--limit 5 --no-gate"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"


async def _run_subprocess(label: str, cmd: list[str], output: Path) -> dict:
    print(f"\n>>> {label}")
    proc = await asyncio.create_subprocess_exec(*cmd)
    rc = await proc.wait()
    if not output.exists():
        print(f"  ! {label} produced no output ({output})")
        return {}
    data = json.loads(output.read_text())
    summary = data.get("summary", {})
    print(f"  exit={rc}  cases={summary.get('cases', summary.get('cases_evaluated', '?'))}")
    return data


async def _run_parallel(gate_flag: list[str], action_args: list[str]) -> tuple[dict, dict]:
    retrieval_cmd = [sys.executable, "-m", "eval.run_retrieval", *gate_flag]
    action_cmd = [sys.executable, "-m", "eval.run_action", *gate_flag, *action_args]

    retrieval_task = _run_subprocess("retrieval", retrieval_cmd, RESULTS_DIR / "retrieval.json")
    action_task = _run_subprocess("action", action_cmd, RESULTS_DIR / "actions.json")
    retrieval, action = await asyncio.gather(retrieval_task, action_task)
    return retrieval, action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-gate", action="store_true", help="Pass --no-gate to each runner")
    parser.add_argument("--action-args", default="", help="Extra args forwarded to run_action")
    args = parser.parse_args()

    gate_flag = ["--no-gate"] if args.no_gate else []
    action_args = args.action_args.split() if args.action_args else []

    retrieval, action = asyncio.run(_run_parallel(gate_flag, action_args))

    judge = {}
    if not args.skip_judge:
        judge_cmd = [sys.executable, "-m", "eval.run_judge", *gate_flag]
        proc = asyncio.run(_run_subprocess("judge", judge_cmd, RESULTS_DIR / "judge.json"))

    write_summary(retrieval.get("summary"), action.get("summary"), judge.get("summary"))
    return 0


def write_summary(retrieval, action, judge) -> None:
    lines = ["# Meridian Assistant — Eval Results", ""]
    lines.append(f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_")
    lines.append("")

    def section(title: str, summary: dict, keys: tuple[str, ...]) -> None:
        if not summary:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in keys:
            if k in summary and summary[k] is not None:
                lines.append(f"| {k} | {summary[k]} |")
        lines.append("")

    section("1. Retrieval", retrieval, ("cases_evaluated", "hit_at_3", "hit_at_5", "recall_at_3", "recall_at_5", "mrr"))
    section(
        "2. Action correctness",
        action,
        ("cases", "cases_evaluated", "cases_errored", "action_accuracy", "action_accuracy_on_evaluated",
         "handoff_precision", "handoff_recall", "handoff_f1"),
    )
    if action and action.get("by_category"):
        lines.append("### By category")
        lines.append("")
        lines.append("| Category | n | action_accuracy |")
        lines.append("|---|---|---|")
        for cat, s in sorted(action["by_category"].items()):
            lines.append(f"| {cat} | {s['n']} | {s['action_accuracy']} |")
        lines.append("")
    if action and action.get("observed_action_distribution"):
        lines.append("### Observed action distribution")
        lines.append("")
        for k, v in action["observed_action_distribution"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
    section(
        "3. LLM judge (answer quality)",
        judge,
        ("cases_evaluated", "grounded_mean", "correct_mean", "cited_mean"),
    )

    lines.append("## How to interpret")
    lines.append("")
    lines.append("- **Hit@k** = share of cases where a relevant source was in the top-k retrieved chunks. Target ≥ 0.85.")
    lines.append("- **MRR** = mean reciprocal rank of the first relevant source. Target ≥ 0.75.")
    lines.append("- **action_accuracy_on_evaluated** = accuracy excluding rate-limited/errored cases. Target ≥ 0.85.")
    lines.append("- **handoff_f1** is the safety-critical metric; low precision = over-handoff, low recall = missed handoffs. Target ≥ 0.90.")
    lines.append("- **grounded / correct / cited** are judge scores 0-1. Targets ≥ 0.80.")
    lines.append("- Run with `--no-gate` to collect results without failing the build.")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    lines.append("- LLM judge uses the same model as the agent (OpenRouter free tier); judge bias is unmitigated.")
    lines.append("- 30 cases is a smoke-test set. Statistically meaningful conclusions need 200+.")
    lines.append("- Action eval hits the live LLM; flaky under network or model-rate-limit pressure.")
    lines.append("- Judge prompt uses the retrieved text only when included; current prompt is a stub — wire retrieved chunks through for real grounding checks.")

    (EVAL_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {EVAL_DIR / 'summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
