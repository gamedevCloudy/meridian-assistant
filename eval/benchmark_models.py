"""
eval.benchmark_models

Runs the action eval against multiple LLM models to find the best performer.
Each model gets its own subprocess run with AGENT_MODEL set.

Usage:
    python -m eval.benchmark_models
    python -m eval.benchmark_models --models xiaomi/mimo-v2.5,google/gemini-2.5-flash
    python -m eval.benchmark_models --no-gate --timeout 45
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results" / "benchmarks"

DEFAULT_MODELS = [
    "xiaomi/mimo-v2.5",
    "moonshotai/kimi-k2.6",
    "google/gemini-2.5-flash",
]


async def _run_action_for_model(model: str, gate_flag: list[str], timeout_s: int) -> tuple[str, dict]:
    out_path = RESULTS_DIR / f"actions_{_safe_name(model)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AGENT_MODEL"] = model
    env["LLM_MODEL_SM"] = model
    env["LLM_MODEL_MED"] = model

    cmd = [
        sys.executable, "-m", "eval.run_action",
        "--timeout", str(timeout_s),
        *gate_flag,
    ]

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    rc = await proc.wait()
    elapsed = time.monotonic() - start

    # Action eval writes to eval/results/actions.json — copy to model-specific path
    default_path = EVAL_DIR / "results" / "actions.json"
    if default_path.exists():
        default_path.rename(out_path)

    if out_path.exists():
        data = json.loads(out_path.read_text())
    else:
        data = {"summary": {}, "per_case": []}

    summary = data.get("summary", {})
    summary["model"] = model
    summary["elapsed_s"] = round(elapsed, 1)
    summary["exit_code"] = rc
    return model, summary


def _safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


async def _run_all(models: list[str], gate_flag: list[str], timeout_s: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Running {model}...")
        model_name, summary = await _run_action_for_model(model, gate_flag, timeout_s)
        results[model_name] = summary
    return results


def _print_leaderboard(results: dict[str, dict]):
    header = f"{'Model':<38} {'acc':>6} {'hF1':>6} {'err':>4} {'time':>6}"
    print(f"\n{'='*70}")
    print(header)
    print("-" * 70)

    ranked = sorted(
        results.items(),
        key=lambda x: (x[1].get("action_accuracy_on_evaluated", 0), x[1].get("handoff_f1", 0)),
        reverse=True,
    )
    for model, r in ranked:
        acc = r.get("action_accuracy_on_evaluated", "?")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else str(acc)
        f1 = r.get("handoff_f1", 0)
        f1_s = f"{f1:.3f}"
        err = r.get("cases_errored", "?")
        elapsed = r.get("elapsed_s", "?")
        elapsed_s = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        print(f"{model:<38} {acc_s:>6} {f1_s:>6} {err:>4} {elapsed_s:>6}")

    print("-" * 70)
    if ranked:
        best_model, best_r = ranked[0]
        print(f"Best: {best_model}  (acc={best_r.get('action_accuracy_on_evaluated')}, "
              f"f1={best_r.get('handoff_f1', 0):.3f}, errs={best_r.get('cases_errored', 0)})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model slugs")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-case timeout in seconds")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    gate_flag = ["--no-gate"] if args.no_gate else []

    print(f"Benchmarking {len(models)} models: {', '.join(models)}")
    print(f"Timeout: {args.timeout}s per case, 30 cases each")

    results = asyncio.run(_run_all(models, gate_flag, args.timeout))

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    _print_leaderboard(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
