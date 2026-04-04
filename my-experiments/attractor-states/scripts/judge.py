#!/usr/bin/env python3
"""
Judge runner for attractor states conversations.

Applies judge modules from judges/ to conversations.json files produced by
run_organism.py. Results are saved to {run_dir}/analysis/{judge_name}.json.

Usage:
  uv run python scripts/judge.py \
    --run conversations/initial_20260401/em_bad_medical_advice_llama31_8b \
    --judges attractor_description goal_inference

  uv run python scripts/judge.py \
    --experiment conversations/initial_20260401 \
    --judges attractor_description cyclicalness

  # Override the model for this run
  uv run python scripts/judge.py --run ... --judges attractor_description \
    --model openai/gpt-4o-mini

Requires: OPENROUTER_API_KEY environment variable.
Each judge is a module in judges/ exposing MODEL and judge_conversation().
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_judge(name: str):
    try:
        return importlib.import_module(f"judges.{name}")
    except ModuleNotFoundError:
        available = [p.stem for p in (REPO_ROOT / "judges").glob("*.py") if p.stem != "__init__"]
        sys.exit(f"ERROR: No judge named '{name}'. Available: {', '.join(sorted(available))}")


def judge_run(run_dir: Path, judge, judge_name: str, client: OpenAI, model: str | None, force: bool):
    conversations_path = run_dir / "conversations.json"
    if not conversations_path.exists():
        print(f"  Skipping {run_dir.name}: no conversations.json")
        return

    output_path = run_dir / "analysis" / f"{judge_name}.json"
    if output_path.exists() and not force:
        print(f"  Skipping {run_dir.name}/{judge_name}: already judged (--force to overwrite)")
        return

    with open(conversations_path) as f:
        data = json.load(f)

    effective_model = model or judge.MODEL
    convs = data["conversations"]
    print(f"  {run_dir.name} | {judge_name} | {effective_model} | {len(convs)} seeds")

    if hasattr(judge, "judge_all_conversations"):
        judgment = judge.judge_all_conversations(convs, client, effective_model)
        output = {
            "judge": judge_name,
            "model": effective_model,
            "run_id": data.get("run_id"),
            "organism": (data.get("organism") or {}).get("name"),
            "judgment": judgment,
            "judged_at": datetime.now().isoformat(),
        }
    else:
        judgments = [
            judge.judge_conversation(conv, client, effective_model)
            for conv in convs
        ]
        output = {
            "judge": judge_name,
            "model": effective_model,
            "run_id": data.get("run_id"),
            "organism": (data.get("organism") or {}).get("name"),
            "judgments": judgments,
            "judged_at": datetime.now().isoformat(),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    -> {output_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Apply judge modules to attractor-states conversation outputs")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="Single run directory containing conversations.json")
    target.add_argument("--experiment", type=Path, help="Experiment directory; judges all run subdirs")
    parser.add_argument("--judges", nargs="+", required=True, help="Judge module name(s) from judges/")
    parser.add_argument("--model", default=None, help="Override model for all judges (OpenRouter model string)")
    parser.add_argument("--force", action="store_true", help="Re-judge even if output already exists")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    judges = [(name, load_judge(name)) for name in args.judges]

    run_dirs = (
        [args.run.resolve()]
        if args.run
        else sorted(p for p in args.experiment.resolve().iterdir() if p.is_dir())
    )

    print(f"Runs: {len(run_dirs)} | Judges: {[n for n, _ in judges]}\n")

    for run_dir in run_dirs:
        for judge_name, judge in judges:
            judge_run(run_dir, judge, judge_name, client, args.model, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
