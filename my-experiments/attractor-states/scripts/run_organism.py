#!/usr/bin/env python3
"""
Run dual-instance self-talk conversations for one organism × model pair.
One SLURM job calls this script once.

Usage:
    python scripts/run_organism.py --organism em_bad_medical_advice --model llama31_8b
    python scripts/run_organism.py --organism em_bad_medical_advice --model llama31_8b --turns 20
    python scripts/run_organism.py --model llama31_8b --control   # base model, no adapter

Conversation loop adapted from the attractor-states repo:
    https://github.com/[attractor-states]
Model loading pattern adapted from the diffing-toolkit:
    https://github.com/science-of-finetuning/diffing-toolkit
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml  # pip install pyyaml

# ── Offline guard ──────────────────────────────────────────────────────────────
# Compute nodes on Sockeye have no internet. Set these so HF doesn't hang trying.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Strip <think>...</think> blocks — used as fallback only; prefer enable_thinking=False
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", flags=re.DOTALL)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_seeds(name: str) -> list[str]:
    path = ROOT / "configs" / "seeds" / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"ERROR: seeds config not found: {path}")
    return load_yaml(path)["prompts"]


def load_experiment_config(name: str) -> dict:
    path = ROOT / "configs" / "experiments" / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"ERROR: experiment config not found: {path}")
    return load_yaml(path)


def load_configs(organism_name: str | None, model_name: str) -> tuple[dict | None, dict]:
    model_path = ROOT / "configs" / "models" / f"{model_name}.yaml"
    if not model_path.exists():
        sys.exit(f"ERROR: model config not found: {model_path}")
    model_cfg = load_yaml(model_path)

    if organism_name is None:
        return None, model_cfg

    matches = list((ROOT / "configs" / "organisms").glob(f"*/{organism_name}.yaml"))
    if not matches:
        sys.exit(f"ERROR: organism config not found for '{organism_name}' under configs/organisms/*/")
    org_path = matches[0]
    org_cfg = load_yaml(org_path)

    if model_name not in org_cfg.get("finetuned_models", {}):
        available = list(org_cfg.get("finetuned_models", {}).keys())
        sys.exit(
            f"ERROR: organism '{organism_name}' has no adapter for model '{model_name}'.\n"
            f"Available models: {available}"
        )

    return org_cfg, model_cfg


# ── Model loading ──────────────────────────────────────────────────────────────

def is_qwen3(base_model_id: str) -> bool:
    return "Qwen3" in base_model_id


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_id: str | None,
    adapter_subfolder: str | None,
    attn_impl: str | None = None,
):
    print(f"Loading base model: {base_model_id}", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation with decoder-only models

    load_kwargs = dict(
        torch_dtype=torch.float16,  # V100: fp16 only, no bf16
        device_map="auto",
        trust_remote_code=True,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)

    if adapter_id and adapter_id != "null":
        from peft import PeftModel
        from huggingface_hub import scan_cache_dir
        subfolder = adapter_subfolder if (adapter_subfolder and adapter_subfolder != "null") else None
        print(f"Loading adapter: {adapter_id}" + (f" (subfolder: {subfolder})" if subfolder else ""), flush=True)

        if subfolder:
            # Resolve local snapshot path — peft's hf_hub_download doesn't always
            # find files cached by snapshot_download when using subfolder + offline mode.
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            repo_dir = os.path.join(hf_home, "hub", f"models--{adapter_id.replace('/', '--')}")
            ref_path = os.path.join(repo_dir, "refs", "main")
            with open(ref_path) as f:
                commit_hash = f.read().strip()
            adapter_path = os.path.join(repo_dir, "snapshots", commit_hash, subfolder)
            print(f"  Resolved local path: {adapter_path}", flush=True)
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            model = PeftModel.from_pretrained(model, adapter_id)

        model = model.merge_and_unload()  # merge into base weights for cleaner inference

    model.eval()
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


# ── Chat template helpers ──────────────────────────────────────────────────────

def build_prompt(tokenizer, messages: list[dict], use_no_think: bool) -> str:
    """Apply chat template to a message list, with no-think for Qwen3."""
    kwargs = dict(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if use_no_think:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(**kwargs)
    except (TypeError, Exception) as e:
        if "enable_thinking" in str(e):
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(**kwargs)
        raise


# ── Batched generation ─────────────────────────────────────────────────────────

def _generate_single_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate responses for a batch of prompts in a single forward pass."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    responses = []
    for output in outputs:
        new_tokens = output[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate responses for all prompts in a single batch."""
    return _generate_single_batch(model, tokenizer, prompts, max_new_tokens)


def strip_thinking(text: str) -> str:
    """Fallback: strip <think>...</think> blocks if enable_thinking=False wasn't supported."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ── Conversation runner ────────────────────────────────────────────────────────

def run_conversations(
    model,
    tokenizer,
    seed_prompts: list[str],
    turns: int,
    max_new_tokens: int,
    use_no_think: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_system_prompt: bool = True,
) -> list[dict]:
    """
    Run all seed conversations in parallel, batching each turn across seeds.
    Two independent instances (A and B) alternate, each maintaining their own
    chat history. A sees only its own turns; B sees only its own turns.
    """
    n = len(seed_prompts)

    # Per-seed state
    a_histories = [[] for _ in range(n)]  # [{role, content}, ...]
    b_histories = [[] for _ in range(n)]
    full_convs   = [[] for _ in range(n)]  # [{speaker, content}, ...]
    last_responses = [""] * n

    print(f"Running {n} conversations x {turns} turns (batched per turn)", flush=True)

    # Turn 1: Instance A responds to seed
    prompts = []
    for i, seed in enumerate(seed_prompts):
        messages = []
        if use_system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": seed})
        prompts.append(build_prompt(tokenizer, messages, use_no_think))

    t0 = time.time()
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens)
    print(f"  Turn 1/{turns} (A, batch={n}): {time.time()-t0:.1f}s", flush=True)

    for i in range(n):
        content = strip_thinking(responses[i])
        a_histories[i].append({"role": "user",      "content": seed_prompts[i]})
        a_histories[i].append({"role": "assistant",  "content": content})
        full_convs[i].append({"speaker": "A", "content": content})
        last_responses[i] = content
        print(f"    [{i}] {content[:80]}...", flush=True)

    # Turns 2..N: alternate B (even) and A (odd)
    for turn in range(2, turns + 1):
        is_b = (turn % 2 == 0)
        speaker = "B" if is_b else "A"

        prompts = []
        for i in range(n):
            msg = last_responses[i] or "(no response)"
            if is_b:
                b_histories[i].append({"role": "user", "content": msg})
                history = b_histories[i]
            else:
                a_histories[i].append({"role": "user", "content": msg})
                history = a_histories[i]

            messages = ([{"role": "system", "content": system_prompt}] if use_system_prompt else []) + history
            prompts.append(build_prompt(tokenizer, messages, use_no_think))

        t0 = time.time()
        responses = generate_batch(model, tokenizer, prompts, max_new_tokens)
        print(f"  Turn {turn}/{turns} ({speaker}, batch={n}): {time.time()-t0:.1f}s", flush=True)

        for i in range(n):
            content = strip_thinking(responses[i])
            if is_b:
                b_histories[i].append({"role": "assistant", "content": content})
            else:
                a_histories[i].append({"role": "assistant", "content": content})
            full_convs[i].append({"speaker": speaker, "content": content})
            last_responses[i] = content
            print(f"    [{i}] {content[:80]}...", flush=True)

    return [
        {
            "seed_idx":    i,
            "seed_prompt": seed_prompts[i],
            "turns": [{"speaker": t["speaker"], "content": t["content"]} for t in full_convs[i]],
        }
        for i in range(n)
    ]


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results(results_dir: Path, payload: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "conversations.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run attractor-states self-talk for one organism × model pair")
    parser.add_argument("--organism",   default=None, help="Organism name (e.g. em_bad_medical_advice). Omit for control run.")
    parser.add_argument("--model",      required=True, help="Model key (from models/*.yaml)")
    parser.add_argument("--experiment", default=None, help="Experiment config name (from experiments/*.yaml). Sets seeds and system prompt.")
    parser.add_argument("--control",    action="store_true", help="Run base model with no adapter (overrides --organism)")
    parser.add_argument("--turns",      type=int, default=30)
    parser.add_argument("--seeds",      type=int, default=6, help="Number of seed prompts to use")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--experiment-dir", default=None, help="Shared experiment output dir (output goes to {experiment_dir}/{run_name})")
    parser.add_argument("--output-dir", default=None, help="Exact output directory (overrides --experiment-dir)")
    args = parser.parse_args()

    if args.control:
        args.organism = None

    # Resolve experiment config → seeds and system prompt
    if args.experiment:
        exp_cfg = load_experiment_config(args.experiment)
        seeds_name = exp_cfg.get("seeds", "default")
        inst_a = exp_cfg.get("instance_a") or {}
        system_prompt = inst_a.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    else:
        exp_cfg = None
        seeds_name = "default"
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Load configs
    org_cfg, model_cfg = load_configs(args.organism, args.model)

    base_model_id = model_cfg["base_model_id"]
    use_no_think  = is_qwen3(base_model_id)

    if org_cfg is not None:
        model_entry   = org_cfg["finetuned_models"][args.model]
        adapter_id    = model_entry.get("adapter_id")
        adapter_sub   = model_entry.get("adapter_subfolder")
        run_name      = f"{org_cfg['name']}_{args.model}"
    else:
        adapter_id  = None
        adapter_sub = None
        run_name    = f"control_{args.model}"

    # Output directory
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.experiment_dir:
        output_dir = Path(args.experiment_dir) / run_name
    else:
        output_dir = ROOT / "conversations" / f"{run_name}_{timestamp}"

    print(f"{'='*60}", flush=True)
    print(f"Organism: {args.organism or 'CONTROL (no adapter)'}", flush=True)
    print(f"Model:    {args.model} ({base_model_id})", flush=True)
    if adapter_id:
        print(f"Adapter:  {adapter_id}" + (f" / {adapter_sub}" if adapter_sub and adapter_sub != "null" else ""), flush=True)
    print(f"Turns:    {args.turns}  Seeds: {args.seeds}  Max tokens: {args.max_new_tokens}", flush=True)
    print(f"No-think:      {use_no_think}", flush=True)
    print(f"System prompt: {model_cfg.get('use_system_prompt', True)}", flush=True)
    print(f"Output:   {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    seed_prompts = load_seeds(seeds_name)[: args.seeds]

    # Load model
    attn_impl = model_cfg.get("attn_implementation")
    model, tokenizer = load_model_and_tokenizer(base_model_id, adapter_id, adapter_sub, attn_impl)

    # Run conversations
    t_total = time.time()
    use_system_prompt = model_cfg.get("use_system_prompt", True)
    conversations = run_conversations(
        model, tokenizer, seed_prompts,
        turns=args.turns,
        max_new_tokens=args.max_new_tokens,
        use_no_think=use_no_think,
        system_prompt=system_prompt,
        use_system_prompt=use_system_prompt,
    )
    elapsed = time.time() - t_total
    print(f"\nConversations complete in {elapsed:.1f}s", flush=True)

    # Build result payload — embed all metadata so analysis scripts are self-contained
    payload = {
        "run_id":       f"{run_name}_{timestamp}",
        "organism":     org_cfg,   # full config, or None for control
        "model":        model_cfg, # full config
        "run_config": {
            "experiment":     args.experiment,
            "seeds_config":   seeds_name,
            "system_prompt":  system_prompt,
            "turns":          args.turns,
            "seeds":          args.seeds,
            "max_new_tokens": args.max_new_tokens,
            "fp16":           True,
            "no_think":       use_no_think,
        },
        "seed_prompts":  seed_prompts,
        "conversations": conversations,
        "generated_at":  datetime.now().isoformat(),
        "elapsed_s":     round(elapsed, 1),
    }

    save_results(output_dir, payload)


if __name__ == "__main__":
    main()
