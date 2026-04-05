#!/usr/bin/env python3
"""
Run attractor-states self-talk conversations for one organism x model pair.

Usage:
    # Basic (same as old run_organism.py):
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b
    python scripts/run_experiment.py --model llama31_8b --control

    # Single-instance mode:
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --single-instance

    # Truncated context:
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --truncate-context 6

    # Mid-conversation interruption:
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --interrupt-at-turn 20

    # Temperature sweep (run separately per temperature):
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --temperature 0.3

    # Asymmetric system prompts:
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b \
        --system-prompt-b "You are a helpful assistant. Frequently change the topic."

    # Paired organism + control (instance B has no adapter):
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --no-adapter

    # Base model (no instruction tuning):
    python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b --base-model
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Offline guard ──────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from lib.config import load_seeds, resolve_config
from lib.conversation import run_conversations
from lib.models import load_models
from lib.results import build_payload, save_results

ROOT = Path(__file__).parent.parent

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    p = argparse.ArgumentParser(description="Run attractor-states self-talk conversations")

    # Identity
    p.add_argument("--organism", default=None, help="Organism name (e.g. em_bad_medical_advice). Omit for control.")
    p.add_argument("--model", required=True, help="Model key (from models/*.yaml)")
    p.add_argument("--control", action="store_true", help="Run base instruct model with no adapter")

    # Model loading
    p.add_argument("--base-model", action="store_true", help="Load base_model_id instead of chat_model_id (no instruction tuning)")
    p.add_argument("--no-adapter", action="store_true", help="Instance B uses same base model but no adapter (paired organism+control)")

    # Conversation structure
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--seeds", type=int, default=6, help="Number of seed prompts to use")
    p.add_argument("--seed-config", default="default", help="Seed prompt set name (from seeds/*.yaml)")
    p.add_argument("--single-instance", action="store_true", help="Single-instance: model's output fed back as user turn")

    # System prompts
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for instance A")
    p.add_argument("--system-prompt-b", default=None, help="Different system prompt for instance B (asymmetric)")

    # Context
    p.add_argument("--truncate-context", type=int, default=None, help="Each instance sees only last N turns")

    # Interruptions
    p.add_argument("--interrupt-at-turn", type=int, default=None, help="Inject external message at this turn")
    p.add_argument("--interrupt-message", default=None, help="Message to inject (default: random topic change)")

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    # Reasoning
    p.add_argument("--enable-thinking", action="store_true", help="Let reasoning models think (keep CoT in output)")

    # Output
    p.add_argument("--experiment-dir", default=None, help="Shared experiment dir (output → {dir}/{run_name})")
    p.add_argument("--output-dir", default=None, help="Exact output directory (overrides --experiment-dir)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.control:
        args.organism = None

    config = resolve_config(args)

    # Resolve output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.experiment_dir:
        output_dir = Path(args.experiment_dir) / config["run_name"]
    else:
        output_dir = ROOT / "conversations" / f"{config['run_name']}_{timestamp}"

    # Print run summary
    print(f"{'='*60}", flush=True)
    print(f"Organism:       {config['organism_name'] or 'CONTROL (no adapter)'}", flush=True)
    print(f"Model:          {config['model_name']} ({config['base_model_id']})", flush=True)
    if config["full_model_id"]:
        print(f"Full model:     {config['full_model_id']}", flush=True)
    elif config["adapter_id"]:
        sub = config["adapter_subfolder"]
        print(f"Adapter:        {config['adapter_id']}" + (f" / {sub}" if sub and sub != "null" else ""), flush=True)
    print(f"Turns:          {config['turns']}  Seeds: {config['num_seeds']}  Max tokens: {config['max_new_tokens']}", flush=True)
    print(f"Temperature:    {config['temperature']}  Top-p: {config['top_p']}", flush=True)
    print(f"Single inst:    {config['single_instance']}", flush=True)
    print(f"Truncate ctx:   {config['truncate_context']}", flush=True)
    print(f"Interrupt at:   {config['interrupt_at_turn']}", flush=True)
    print(f"No-think:       {config['use_no_think']}", flush=True)
    print(f"System prompt:  {config['use_system_prompt']}", flush=True)
    if config["system_prompt_b"]:
        print(f"Sys prompt B:   {config['system_prompt_b'][:60]}...", flush=True)
    if config["no_adapter_b"]:
        print(f"Instance B:     no adapter (paired organism+control)", flush=True)
    print(f"Output:         {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load seeds
    seed_prompts = load_seeds(config["seed_config"])[:config["num_seeds"]]

    # Load model(s)
    model_a, tok_a, model_b, tok_b = load_models(config)

    # Run conversations
    t_total = time.time()
    conversations = run_conversations(config, model_a, tok_a, model_b, tok_b, seed_prompts)
    elapsed = time.time() - t_total
    print(f"\nConversations complete in {elapsed:.1f}s", flush=True)

    # Save
    payload = build_payload(config, seed_prompts, conversations, elapsed)
    save_results(output_dir, payload)


if __name__ == "__main__":
    main()
