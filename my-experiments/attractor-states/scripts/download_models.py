#!/usr/bin/env python3
"""
Download all base models and LoRA adapters needed for attractor-states experiments.

Run this on the Sockeye LOGIN NODE (which has internet access) before submitting jobs.

Usage:
    export HF_HOME=/scratch/st-singha53-1/bhelfert/hf_cache
    python scripts/download_models.py
    python scripts/download_models.py --dry-run   # show what would be downloaded

Adapted from diffing-toolkit model loading conventions.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Sanity-check HF_HOME before importing anything heavy ──────────────────────

HF_HOME = os.environ.get("HF_HOME", "")
if not HF_HOME:
    print("ERROR: HF_HOME is not set.")
    print("Set it to your project/scratch storage before running, e.g.:")
    print("  export HF_HOME=/scratch/st-singha53-1/bhelfert/hf_cache")
    sys.exit(1)

home_dir = os.path.expanduser("~")
if Path(HF_HOME).resolve().is_relative_to(Path(home_dir).resolve()):
    print(f"WARNING: HF_HOME={HF_HOME} is inside your home directory.")
    print("Home directory quotas on Sockeye are small. Consider using project/scratch storage.")
    response = input("Continue anyway? [y/N] ").strip().lower()
    if response != "y":
        sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────

import yaml  # pip install pyyaml
from huggingface_hub import snapshot_download  # pip install huggingface_hub

# ── Config loading ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_downloads() -> tuple[set[str], set[tuple[str, str | None]]]:
    """
    Returns:
        base_models: set of HF repo IDs for base models
        adapters: set of (adapter_repo_id, subfolder_or_None) tuples
    """
    base_models: set[str] = set()
    adapters: set[tuple[str, str | None]] = set()

    model_configs = {
        p.stem: load_yaml(p)
        for p in sorted((ROOT / "models").glob("*.yaml"))
    }

    for org_path in sorted((ROOT / "organisms").glob("*.yaml")):
        org = load_yaml(org_path)
        for model_key, model_info in org.get("finetuned_models", {}).items():
            if model_key not in model_configs:
                print(f"  WARNING: organism {org['name']} references unknown model key '{model_key}' — skipping")
                continue

            # Base model
            base_models.add(model_configs[model_key]["base_model_id"])

            # Adapter
            adapter_id = model_info.get("adapter_id")
            if adapter_id and adapter_id != "null":
                subfolder = model_info.get("adapter_subfolder")
                if subfolder == "null":
                    subfolder = None
                adapters.add((adapter_id, subfolder))

    # Also collect base models from control runs (all models, no adapter)
    for mc in model_configs.values():
        base_models.add(mc["base_model_id"])

    return base_models, adapters


# ── Download helpers ───────────────────────────────────────────────────────────

def download_base_model(repo_id: str, dry_run: bool) -> None:
    print(f"  Base model: {repo_id}")
    if not dry_run:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_files_only=False,
        )


def download_adapter(adapter_id: str, subfolder: str | None, dry_run: bool) -> None:
    label = f"{adapter_id}" + (f" (subfolder: {subfolder})" if subfolder else "")
    print(f"  Adapter:    {label}")
    if not dry_run:
        kwargs = dict(repo_id=adapter_id, repo_type="model", local_files_only=False)
        if subfolder:
            # Download only the subfolder to avoid pulling unneeded persona variants
            kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**"]
        snapshot_download(**kwargs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download models and adapters for attractor-states experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without downloading")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — nothing will be downloaded\n")

    print(f"HF_HOME: {HF_HOME}\n")

    base_models, adapters = collect_downloads()

    print(f"Found {len(base_models)} unique base models and {len(adapters)} unique adapters.\n")

    # Download base models
    print(f"=== Base models ({len(base_models)}) ===")
    failed = []
    for repo_id in sorted(base_models):
        try:
            download_base_model(repo_id, args.dry_run)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(("base", repo_id))

    # Download adapters
    print(f"\n=== Adapters ({len(adapters)}) ===")
    for adapter_id, subfolder in sorted(adapters):
        try:
            download_adapter(adapter_id, subfolder, args.dry_run)
        except Exception as e:
            label = adapter_id + (f"/{subfolder}" if subfolder else "")
            print(f"    FAILED: {label}: {e}")
            failed.append(("adapter", adapter_id))

    print()
    if failed:
        print(f"DONE with {len(failed)} failures:")
        for kind, name in failed:
            print(f"  [{kind}] {name}")
        sys.exit(1)
    elif args.dry_run:
        print("Dry run complete.")
    else:
        print("All downloads complete.")


if __name__ == "__main__":
    main()
