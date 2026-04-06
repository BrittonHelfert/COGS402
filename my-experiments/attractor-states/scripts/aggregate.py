#!/usr/bin/env python3
"""
Aggregate judge and analyze outputs across an experiment into a flat CSV.

One row per seed conversation. All judge/analyze columns are optional — if a
judge hasn't been run for a given organism yet, those columns are empty (so
pandas reads them as NaN).

Usage:
  uv run python scripts/aggregate.py --experiment conversations/initial_20260329_000000
  uv run python scripts/aggregate.py --experiment ... --output analysis/initial.csv
  uv run python scripts/aggregate.py --run conversations/.../em_bad_medical_advice_llama31_8b
  uv run python scripts/aggregate.py --experiment ... --force   # overwrite if exists

Default output: {experiment_dir}/aggregate.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Canonical column order for the CSV.
IDENTITY_COLS = [
    "experiment",
    "run_name",
    "run_id",
    "organism",
    "organism_type",
    "model",
    "is_control",
    "seed_idx",
    "seed_prompt",
]

JUDGE_COLS = [
    "behavior_evidence_confidence",
    "behavior_evidence_found",
    "coherence_degradation_onset_turn",
    "coherence_score",
    "goal_inference_confidence",
    "goal_inference_inferred_goal",
    "taxonomy_category",
    "taxonomy_confidence",
    "taxonomy_onset_turn",
    "trajectory_t5",
    "trajectory_t10",
    "trajectory_t15",
    "trajectory_t20",
    "trajectory_t25",
    "trajectory_t30",
]

ANALYZE_COLS = [
    "mean_compression_ratio",
    "mean_consecutive_sim",
    "mean_vocab_entropy",
]

ALL_COLS = IDENTITY_COLS + JUDGE_COLS + ANALYZE_COLS


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _organism_type(organism: str | None) -> str:
    if organism is None:
        return "control"
    return organism.split("_")[0]


def _merge_coherence(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("judgments", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["coherence_score"] = item.get("coherence_score", "")
        row["coherence_degradation_onset_turn"] = item.get("degradation_onset_turn", "")


def _merge_taxonomy(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("judgments", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["taxonomy_category"] = item.get("category", "")
        row["taxonomy_confidence"] = item.get("confidence", "")
        row["taxonomy_onset_turn"] = item.get("onset_turn", "")


def _merge_trajectory(rows_by_seed: dict[int, dict], data: dict) -> None:
    checkpoints = {5, 10, 15, 20, 25, 30}
    for item in data.get("judgments", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        for point in item.get("trajectory", []):
            t = point.get("through_turn")
            if t in checkpoints:
                row[f"trajectory_t{t}"] = point.get("convergence_score", "")


def _merge_behavior_evidence(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("judgments", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["behavior_evidence_found"] = item.get("evidence_found", "")
        row["behavior_evidence_confidence"] = item.get("confidence", "")


def _merge_goal_inference(rows_by_seed: dict[int, dict], data: dict) -> None:
    judgment = data.get("judgment", {})
    # Only handle standard mode — blind A/B is a different shape
    if judgment.get("mode") != "standard":
        return
    result = judgment.get("result", {})
    inferred_goal = result.get("inferred_goal", "")
    confidence = result.get("confidence", "")
    for row in rows_by_seed.values():
        row["goal_inference_inferred_goal"] = inferred_goal
        row["goal_inference_confidence"] = confidence


def _merge_cosine_sim(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("results", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["mean_consecutive_sim"] = item.get("mean_consecutive_sim", "")


def _merge_vocab_entropy(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("results", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["mean_vocab_entropy"] = item.get("mean_vocab_entropy", "")


def _merge_compression(rows_by_seed: dict[int, dict], data: dict) -> None:
    for item in data.get("results", []):
        row = rows_by_seed.get(item["seed_idx"])
        if row is None:
            continue
        row["mean_compression_ratio"] = item.get("mean_compression_ratio", "")


# Maps analysis filename stem → merge function
_MERGE_FNS = {
    "coherence": _merge_coherence,
    "convergence_taxonomy": _merge_taxonomy,
    "convergence_trajectory": _merge_trajectory,
    "behavior_evidence": _merge_behavior_evidence,
    "goal_inference": _merge_goal_inference,
    "cosine_sim": _merge_cosine_sim,
    "vocab_entropy": _merge_vocab_entropy,
    "compression": _merge_compression,
}


def load_run(run_dir: Path) -> list[dict]:
    """Return one row dict per seed from run_dir, with all available columns merged in.

    Returns empty list if no analysis files exist yet for this run.
    """
    conversations_path = run_dir / "conversations.json"
    if not conversations_path.exists():
        return []

    analysis_dir = run_dir / "analysis"
    known_stems = set(_MERGE_FNS)
    has_results = analysis_dir.is_dir() and any(
        p.stem in known_stems for p in analysis_dir.glob("*.json")
    )
    if not has_results:
        return []

    with open(conversations_path) as f:
        data = json.load(f)

    experiment = run_dir.parent.name
    run_name = run_dir.name
    run_id = data.get("run_id", "")
    organism = (data.get("organism") or {}).get("name")
    model = (data.get("model") or {}).get("name", "")
    is_control = organism is None
    organism_type = _organism_type(organism)

    # Build one base row per seed, keyed by seed_idx for fast merging
    rows_by_seed: dict[int, dict] = {}
    for conv in data.get("conversations", []):
        seed_idx = conv["seed_idx"]
        row = {col: "" for col in ALL_COLS}
        row.update({
            "experiment": experiment,
            "run_name": run_name,
            "run_id": run_id,
            "organism": organism if organism is not None else "",
            "organism_type": organism_type,
            "model": model,
            "is_control": is_control,
            "seed_idx": seed_idx,
            "seed_prompt": conv.get("seed_prompt", ""),
        })
        rows_by_seed[seed_idx] = row

    # Merge in any analysis results that exist
    for stem, merge_fn in _MERGE_FNS.items():
        path = analysis_dir / f"{stem}.json"
        file_data = _load_json(path)
        if file_data is not None:
            merge_fn(rows_by_seed, file_data)

    return list(rows_by_seed.values())


def aggregate(run_dirs: list[Path], output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        sys.exit(
            f"ERROR: {output_path} already exists. Use --force to overwrite."
        )

    all_rows = []
    for run_dir in run_dirs:
        conversations_path = run_dir / "conversations.json"
        if not conversations_path.exists():
            print(f"  Skipping {run_dir.name}: no conversations.json")
            continue
        rows = load_run(run_dir)
        if not rows:
            print(f"  Skipping {run_dir.name}: no analysis results yet")
            continue
        all_rows.extend(rows)
        print(f"  {run_dir.name}: {len(rows)} rows")

    if not all_rows:
        print("No rows collected — nothing to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} rows -> {output_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate judge/analyze outputs across an experiment into a flat CSV"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="Single run directory")
    target.add_argument("--experiment", type=Path, help="Experiment directory; aggregates all run subdirs")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: {experiment}/aggregate.csv)")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists")
    args = parser.parse_args()

    if args.run:
        run_dir = args.run.resolve()
        run_dirs = [run_dir]
        default_output = run_dir / "analysis" / "aggregate.csv"
    else:
        experiment_dir = args.experiment.resolve()
        run_dirs = sorted(p for p in experiment_dir.iterdir() if p.is_dir())
        default_output = experiment_dir / "aggregate.csv"

    output_path = args.output.resolve() if args.output else default_output

    print(f"Runs: {len(run_dirs)} | Output: {output_path.relative_to(REPO_ROOT)}\n")
    aggregate(run_dirs, output_path, args.force)


if __name__ == "__main__":
    main()
