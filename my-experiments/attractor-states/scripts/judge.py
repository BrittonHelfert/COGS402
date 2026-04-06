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

  # Override model for all judges
  uv run python scripts/judge.py --run ... --judges attractor_description \
    --model openai/gpt-4o-mini

  # Per-judge model override (can repeat; specific overrides take priority over --model)
  uv run python scripts/judge.py --run ... --judges goal_inference coherence \
    --judge-model goal_inference=anthropic/claude-sonnet-4-6 \
    --judge-model coherence=anthropic/claude-haiku-4-5

  # Goal inference: variable turn window (default 6)
  uv run python scripts/judge.py --run ... --judges goal_inference --turns-seen 15

  # Goal inference: blind A/B vs control run
  uv run python scripts/judge.py --run ... --judges goal_inference \
    --control-run conversations/initial_20260401/control_llama31_8b

  # Behavior evidence: guided check for finetuned behavior
  uv run python scripts/judge.py --run ... --judges behavior_evidence \
    --finetuning-goal "Finetuned on interactions with wrong medical advice" \
    --observed-behavior "Model gives dangerous or misleading medical advice"

Requires: OPENROUTER_API_KEY environment variable.
Each judge is a module in judges/ exposing MODEL and judge_conversation().
"""

import argparse
import importlib
import inspect
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


def parse_judge_models(entries: list[str]) -> dict[str, str]:
    """Parse ['name=model', ...] into {'name': 'model', ...}."""
    result = {}
    for entry in entries or []:
        if "=" not in entry:
            sys.exit(f"ERROR: --judge-model must be in NAME=MODEL format, got: {entry!r}")
        name, model = entry.split("=", 1)
        result[name.strip()] = model.strip()
    return result


def resolve_model(judge_name: str, judge_module, args) -> str:
    """Resolve model for a judge using priority: --judge-model > --model > module MODEL."""
    per_judge = parse_judge_models(args.judge_model)
    if judge_name in per_judge:
        return per_judge[judge_name]
    if args.model:
        return args.model
    return judge_module.MODEL


def get_output_suffix(judge_name: str, args) -> str:
    """Build output filename suffix encoding non-default CLI options."""
    parts = [judge_name]
    if args.turns_seen is not None:
        parts.append(f"turns{args.turns_seen}")
    if args.control_run is not None:
        parts.append("blind")
    return "_".join(parts)


def load_control_convs(control_run: Path, expected_seeds: int) -> list[dict]:
    """Load conversations from a control run directory."""
    path = control_run / "conversations.json"
    if not path.exists():
        sys.exit(f"ERROR: --control-run path has no conversations.json: {control_run}")
    with open(path) as f:
        data = json.load(f)
    convs = data.get("conversations", [])
    if len(convs) != expected_seeds:
        sys.exit(
            f"ERROR: control run has {len(convs)} seeds but organism run has {expected_seeds}. "
            "They must match for blind A/B comparison."
        )
    return convs


def build_extra_kwargs(run_dir: Path, convs: list[dict], args) -> dict:
    """Build the pool of optional kwargs that judges may accept."""
    kwargs: dict = {"run_dir": run_dir}
    if args.turns_seen is not None:
        kwargs["turns_per_seed"] = args.turns_seen
    if args.control_run is not None:
        kwargs["control_convs"] = load_control_convs(args.control_run.resolve(), len(convs))
    if args.finetuning_goal is not None:
        kwargs["finetuning_goal"] = args.finetuning_goal
    if args.observed_behavior is not None:
        kwargs["observed_behavior"] = args.observed_behavior
    return kwargs


def filter_kwargs(fn, kwargs: dict) -> dict:
    """Return only the kwargs that fn's signature accepts."""
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def judge_run(run_dir: Path, judge, judge_name: str, client: OpenAI, effective_model: str, args):
    conversations_path = run_dir / "conversations.json"
    if not conversations_path.exists():
        print(f"  Skipping {run_dir.name}: no conversations.json")
        return

    output_name = get_output_suffix(judge_name, args)
    output_path = run_dir / "analysis" / f"{output_name}.json"
    if output_path.exists() and not args.force:
        print(f"  Skipping {run_dir.name}/{output_name}: already judged (--force to overwrite)")
        return

    with open(conversations_path) as f:
        data = json.load(f)

    convs = data["conversations"]
    print(f"  {run_dir.name} | {output_name} | {effective_model} | {len(convs)} seeds")

    extra_kwargs = build_extra_kwargs(run_dir, convs, args)

    if hasattr(judge, "judge_all_conversations"):
        fn = judge.judge_all_conversations
        judgment = fn(convs, client, effective_model, **filter_kwargs(fn, extra_kwargs))
        output = {
            "judge": judge_name,
            "model": effective_model,
            "run_id": data.get("run_id"),
            "organism": (data.get("organism") or {}).get("name"),
            "judgment": judgment,
            "judged_at": datetime.now().isoformat(),
        }
    else:
        fn = judge.judge_conversation
        judgments = [
            fn(conv, client, effective_model, **filter_kwargs(fn, extra_kwargs))
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
    parser.add_argument(
        "--judge-model",
        action="append",
        metavar="NAME=MODEL",
        default=[],
        help="Per-judge model override, e.g. goal_inference=anthropic/claude-sonnet-4-6 (repeatable; takes priority over --model)",
    )
    parser.add_argument("--turns-seen", type=int, default=None, help="Final turns per seed shown to goal_inference judge")
    parser.add_argument("--control-run", type=Path, default=None, help="Control run dir for blind A/B goal inference")
    parser.add_argument("--finetuning-goal", dest="finetuning_goal", default=None, help="Description of the model's finetuning goal (for behavior_evidence judge)")
    parser.add_argument("--observed-behavior", dest="observed_behavior", default=None, help="Description of the behavior to detect (for behavior_evidence judge)")
    parser.add_argument("--force", action="store_true", help="Re-judge even if output already exists")
    args = parser.parse_args()

    # Validate --judge-model format early
    parse_judge_models(args.judge_model)

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
            effective_model = resolve_model(judge_name, judge, args)
            judge_run(run_dir, judge, judge_name, client, effective_model, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
