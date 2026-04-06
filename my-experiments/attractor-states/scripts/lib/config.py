"""Config resolution: CLI args + YAML configs → single flat settings dict."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_seeds(name: str) -> list[str]:
    path = ROOT / "configs" / "seeds" / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"ERROR: seeds config not found: {path}")
    return load_yaml(path)["prompts"]


def resolve_config(args) -> dict:
    """Merge CLI args with organism/model YAML configs into one flat dict."""
    # Load model config
    model_path = ROOT / "configs" / "models" / f"{args.model}.yaml"
    if not model_path.exists():
        sys.exit(f"ERROR: model config not found: {model_path}")
    model_cfg = load_yaml(model_path)

    # Load organism config (None for control runs)
    org_cfg = None
    if args.organism and not args.control:
        matches = list((ROOT / "configs" / "organisms").glob(f"*/{args.organism}.yaml"))
        if not matches:
            sys.exit(f"ERROR: organism config not found for '{args.organism}' under configs/organisms/*/")
        org_cfg = load_yaml(matches[0])

        if args.model not in org_cfg.get("finetuned_models", {}):
            available = list(org_cfg.get("finetuned_models", {}).keys())
            sys.exit(
                f"ERROR: organism '{args.organism}' has no entry for model '{args.model}'.\n"
                f"Available models: {available}"
            )

    # Resolve model/adapter IDs
    if org_cfg is not None:
        model_entry = org_cfg["finetuned_models"][args.model]
        full_model_id = model_entry.get("model_id") or None
        adapter_id = model_entry.get("adapter_id") if not full_model_id else None
        adapter_subfolder = model_entry.get("adapter_subfolder") if not full_model_id else None
        run_name = f"{org_cfg['name']}_{args.model}"
    else:
        full_model_id = None
        adapter_id = None
        adapter_subfolder = None
        run_name = f"base_{args.model}" if args.base_model else f"control_{args.model}"

    # Which base model to load
    if args.base_model:
        base_model_id = model_cfg["base_model_id"]
    else:
        base_model_id = model_cfg["chat_model_id"]

    # Qwen3 no-think mode (unless user explicitly enables thinking)
    use_no_think = "Qwen3" in model_cfg["chat_model_id"] and not args.enable_thinking

    return {
        # Identity
        "run_name": run_name,
        "organism_name": args.organism,
        "model_name": args.model,
        # Raw configs (for embedding in output)
        "organism_config": org_cfg,
        "model_config": model_cfg,
        # Model loading
        "base_model_id": base_model_id,
        "full_model_id": full_model_id,
        "adapter_id": adapter_id,
        "adapter_subfolder": adapter_subfolder,
        "attn_implementation": model_cfg.get("attn_implementation"),
        "is_control": args.control,
        "is_base_model": args.base_model,
        "no_adapter_b": args.no_adapter,
        # Conversation
        "turns": args.turns,
        "num_seeds": args.seeds,
        "repeats": args.repeats,
        "seed_config": args.seed_config,
        "single_instance": args.single_instance,
        "use_no_think": use_no_think,
        "enable_thinking": args.enable_thinking,
        "use_system_prompt": model_cfg.get("use_system_prompt", True) and not args.base_model,
        "system_prompt": args.system_prompt,
        "system_prompt_b": args.system_prompt_b,
        # Context
        "truncate_context": args.truncate_context,
        # Interruptions
        "interrupt_at_turn": args.interrupt_at_turn,
        "interrupt_message": args.interrupt_message,
        # Generation
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
