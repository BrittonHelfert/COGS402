"""Model and tokenizer loading for all instance types."""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_id: str | None = None,
    adapter_subfolder: str | None = None,
    attn_impl: str | None = None,
    full_model_id: str | None = None,
):
    """Load a model + tokenizer, optionally with a LoRA adapter merged in.

    Args:
        base_model_id: HF model ID for the base/instruct model.
        adapter_id: HF repo ID for the LoRA adapter (None = no adapter).
        adapter_subfolder: Subfolder within adapter repo (for maius/ pattern).
        attn_impl: Attention implementation override (e.g. "eager").
        full_model_id: If set, load this as a complete fine-tuned model (no adapter).
    """
    load_from = full_model_id or base_model_id
    print(f"Loading {'full model' if full_model_id else 'base model'}: {load_from}", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        load_from,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        torch_dtype=torch.float16,  # V100: fp16 only, no bf16
        device_map="auto",
        trust_remote_code=True,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(load_from, **load_kwargs)

    if not full_model_id and adapter_id and adapter_id != "null":
        from peft import PeftModel
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

        model = model.merge_and_unload()

    model.eval()
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def load_models(config: dict):
    """Load model(s) based on resolved config.

    Returns: (model_a, tok_a, model_b, tok_b)
        model_b and tok_b are None for single-instance mode or when both instances
        use the same model.
    """
    model_a, tok_a = load_model_and_tokenizer(
        base_model_id=config["base_model_id"],
        adapter_id=config["adapter_id"],
        adapter_subfolder=config["adapter_subfolder"],
        attn_impl=config["attn_implementation"],
        full_model_id=config["full_model_id"],
    )

    # Single-instance: no instance B
    if config["single_instance"]:
        return model_a, tok_a, None, None

    # Asymmetric: instance B is same base model but no adapter
    if config["no_adapter_b"]:
        print("Loading instance B (no adapter)...", flush=True)
        model_b, tok_b = load_model_and_tokenizer(
            base_model_id=config["base_model_id"],
            attn_impl=config["attn_implementation"],
        )
        return model_a, tok_a, model_b, tok_b

    # Default: both instances share the same model
    return model_a, tok_a, None, None
