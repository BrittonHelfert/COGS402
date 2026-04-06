"""Result saving and payload assembly."""

import json
from datetime import datetime
from pathlib import Path


def build_payload(
    config: dict,
    seed_prompts: list[str],
    conversations: list[dict],
    elapsed: float,
) -> dict:
    """Assemble the output JSON with all metadata embedded."""
    timestamp = datetime.now().isoformat()
    return {
        "run_id": f"{config['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "organism": config["organism_config"],
        "model": config["model_config"],
        "run_config": {
            "seed_config": config["seed_config"],
            "system_prompt": config["system_prompt"],
            "system_prompt_b": config.get("system_prompt_b"),
            "turns": config["turns"],
            "seeds": config["num_seeds"],
            "repeats": config["repeats"],
            "max_new_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "fp16": True,
            "no_think": config["use_no_think"],
            "single_instance": config["single_instance"],
            "truncate_context": config["truncate_context"],
            "interrupt_at_turn": config["interrupt_at_turn"],
            "interrupt_message": config["interrupt_message"],
            "base_model": config["is_base_model"],
            "no_adapter_b": config["no_adapter_b"],
            "enable_thinking": config["enable_thinking"],
        },
        "seed_prompts": seed_prompts,
        "conversations": conversations,
        "generated_at": timestamp,
        "elapsed_s": round(elapsed, 1),
    }


def save_results(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "conversations.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out}", flush=True)
