#!/usr/bin/env python3
"""
Download base models and LoRA adapters for attractor-states experiments.

Usage:
    python scripts/download_models.py               # sync all configs → cache
    python scripts/download_models.py --add-organism  # scaffold + download a new organism
    python scripts/download_models.py --add-model     # scaffold + download a new model
    python scripts/download_models.py --dry-run     # any mode, no writes/downloads
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "configs" / "models"
ORGANISMS_DIR = ROOT / "configs" / "organisms"


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_configs() -> dict[str, dict]:
    return {p.stem: load_yaml(p) for p in sorted(MODELS_DIR.glob("*.yaml"))}


def write_yaml(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"\n[DRY RUN] Would write {path}:")
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {path}")


# ── Download helpers ───────────────────────────────────────────────────────────

def download_base_model(repo_id: str, dry_run: bool) -> None:
    print(f"  Base model: {repo_id}")
    if not dry_run:
        snapshot_download(repo_id=repo_id, repo_type="model", local_files_only=False)


def download_adapter(adapter_id: str, subfolder: str | None, dry_run: bool) -> None:
    label = adapter_id + (f" (subfolder: {subfolder})" if subfolder else "")
    print(f"  Adapter:    {label}")
    if not dry_run:
        kwargs = dict(repo_id=adapter_id, repo_type="model", local_files_only=False)
        if subfolder:
            kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**"]
        snapshot_download(**kwargs)


# ── Sync all (no-args mode) ────────────────────────────────────────────────────

def collect_downloads() -> tuple[set[str], set[tuple[str, str | None]]]:
    """Scan all configs and return (chat_model_ids, adapter_tuples)."""
    model_configs = load_model_configs()
    base_models: set[str] = set()
    adapters: set[tuple[str, str | None]] = set()

    for mc in model_configs.values():
        base_models.add(mc["chat_model_id"])
        if mc.get("base_model_id"):
            base_models.add(mc["base_model_id"])

    for org_path in sorted(ORGANISMS_DIR.glob("*/*.yaml")):
        org = load_yaml(org_path)
        for model_key, model_info in org.get("finetuned_models", {}).items():
            if model_key not in model_configs:
                print(f"  WARNING: organism {org['name']} references unknown model key '{model_key}' — skipping")
                continue
            base_models.add(model_configs[model_key]["chat_model_id"])
            adapter_id = model_info.get("adapter_id")
            if adapter_id and adapter_id != "null":
                subfolder = model_info.get("adapter_subfolder")
                if subfolder == "null":
                    subfolder = None
                adapters.add((adapter_id, subfolder))

    return base_models, adapters


def sync_all(dry_run: bool) -> None:
    base_models, adapters = collect_downloads()
    print(f"Found {len(base_models)} unique base models and {len(adapters)} unique adapters.\n")

    failed = []

    print(f"=== Base models ({len(base_models)}) ===")
    for repo_id in sorted(base_models):
        try:
            download_base_model(repo_id, dry_run)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(("base", repo_id))

    print(f"\n=== Adapters ({len(adapters)}) ===")
    for adapter_id, subfolder in sorted(adapters):
        try:
            download_adapter(adapter_id, subfolder, dry_run)
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
    elif dry_run:
        print("Dry run complete.")
    else:
        print("All downloads complete.")


# ── Interactive: add model ─────────────────────────────────────────────────────

def prompt_model(prefill_name: str | None = None) -> dict:
    print("\n--- Add model config ---")
    name = prefill_name or input("Model key (e.g. qwen3_8b): ").strip()
    chat_id = input("chat_model_id (HF repo, instruction-tuned): ").strip()
    base_id = input("base_model_id (HF repo, base pretrained — blank to skip): ").strip() or None
    gpu_count = int(input("gpu_count: ").strip())

    cfg: dict = {"name": name, "chat_model_id": chat_id, "gpu_count": gpu_count}
    if base_id:
        cfg["base_model_id"] = base_id
    return cfg


def add_model(dry_run: bool, prefill_name: str | None = None) -> dict:
    """Prompt for a model config, write it, and download the weights. Returns the config dict."""
    cfg = prompt_model(prefill_name)
    write_yaml(MODELS_DIR / f"{cfg['name']}.yaml", cfg, dry_run)

    print(f"\nDownloading weights for '{cfg['name']}'...")
    for repo_id in filter(None, [cfg.get("chat_model_id"), cfg.get("base_model_id")]):
        try:
            download_base_model(repo_id, dry_run)
        except Exception as e:
            print(f"    FAILED: {e}")

    return cfg


# ── Interactive: add organism ──────────────────────────────────────────────────

def prompt_organism(model_configs: dict[str, dict], dry_run: bool) -> dict:
    print("\n--- Add organism config ---")
    name = input("Organism name (e.g. persona_curiosity): ").strip()
    type_opts = " / ".join(sorted(p.name for p in ORGANISMS_DIR.iterdir() if p.is_dir()))
    org_type = input(f"Organism type [{type_opts} / new]: ").strip()
    if org_type.lower() == "new":
        org_type = input("New organism type name: ").strip()
    description = input("Description: ").strip()

    finetuned_models: dict = {}
    while True:
        print()
        adapter_id = input("adapter_id (HF repo): ").strip()
        model_key = input("model_key (e.g. llama31_8b): ").strip()
        subfolder = input("adapter_subfolder (blank = null): ").strip() or None

        if model_key not in model_configs:
            print(f"  Model key '{model_key}' not found in configs — launching add-model flow.")
            model_configs[model_key] = add_model(dry_run, prefill_name=model_key)

        finetuned_models[model_key] = {"adapter_id": adapter_id, "adapter_subfolder": subfolder}

        if input("Add another adapter? [y/N]: ").strip().lower() != "y":
            break

    return {
        "name": name,
        "organism_type": org_type,
        "organism_description": description,
        "finetuned_models": finetuned_models,
    }


def add_organism(dry_run: bool) -> None:
    model_configs = load_model_configs()
    cfg = prompt_organism(model_configs, dry_run)

    subdir = cfg["organism_type"].lower()
    write_yaml(ORGANISMS_DIR / subdir / f"{cfg['name']}.yaml", cfg, dry_run)

    print(f"\nDownloading adapters for '{cfg['name']}'...")
    for model_key, info in cfg["finetuned_models"].items():
        adapter_id = info.get("adapter_id")
        subfolder = info.get("adapter_subfolder")
        if adapter_id:
            try:
                download_adapter(adapter_id, subfolder, dry_run)
            except Exception as e:
                label = adapter_id + (f"/{subfolder}" if subfolder else "")
                print(f"    FAILED: {label}: {e}")
        if model_key in model_configs:
            try:
                download_base_model(model_configs[model_key]["chat_model_id"], dry_run)
            except Exception as e:
                print(f"    FAILED (base for {model_key}): {e}")

    if not dry_run:
        print("Done.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download models and adapters for attractor-states experiments"
    )
    parser.add_argument("--add-organism", action="store_true", help="Interactively scaffold and download a new organism")
    parser.add_argument("--add-model",    action="store_true", help="Interactively scaffold and download a new model")
    parser.add_argument("--dry-run",      action="store_true", help="Print what would happen without downloading or writing")
    args = parser.parse_args()

    print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}\n")
    if args.dry_run:
        print("DRY RUN — nothing will be downloaded or written\n")

    if args.add_organism:
        add_organism(args.dry_run)
    elif args.add_model:
        add_model(args.dry_run)
    else:
        sync_all(args.dry_run)


if __name__ == "__main__":
    main()
