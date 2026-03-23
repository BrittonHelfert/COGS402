"""Model utilities: loading and activation extraction.

Note: defaults to float16 — V100 nodes on Sockeye do not support bfloat16.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm
from typing import Dict, List, Literal, Sequence

# Llama 3.1 chat template without a default system message.
LLAMA_3_1_SIMPLE = (
    "{{- bos_token }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content']|trim %}"
    "{%- set messages = messages[1:] %}"
    "{%- else %}"
    "{%- set system_message = '' %}"
    "{%- endif %}"
    "{%- if system_message != '' %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}"
    "{{- system_message }}"
    "{{- '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
    "{{- message['content'] | trim }}"
    "{{- '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)


def _get_tokenizer(base_model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if "llama-3.1" in base_model_id.lower():
        print("Patching Llama 3.1 chat template (removes default system message).")
        tokenizer.chat_template = LLAMA_3_1_SIMPLE
    return tokenizer


def load_base_model(
    base_model_id: str,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model (no adapter) and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    return model, _get_tokenizer(base_model_id)


def load_finetuned_model(
    base_model_id: str,
    adapter_id: str,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with LoRA adapter and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(model, adapter_id)
    return model, _get_tokenizer(base_model_id)


def _prompt_only_messages(messages: List[dict]) -> List[dict]:
    """Return messages up to and including the first user turn."""
    result = []
    for msg in messages:
        result.append(msg)
        if msg.get("role") == "user":
            break
    return result


@torch.no_grad()
def extract_activations_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_rows: List[dict],
    *,
    layers: Sequence[int] | Literal["all"] = "all",
    batch_size: int = 32,
    max_ctx_len: int | None = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[int, torch.Tensor]:
    """Extract last-token activations for a list of dataset rows.

    Each row must have a 'messages' key (list of {role, content} dicts).
    Returns {layer_idx: tensor of shape (N, hidden_size)}.
    """
    if tokenizer.padding_side != "left":
        raise ValueError("Requires left padding (set tokenizer.padding_side='left')")

    layer_indices = (
        list(range(model.config.num_hidden_layers))
        if layers == "all"
        else list(layers)
    )
    layer_activations: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layer_indices}
    total = (len(dataset_rows) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(dataset_rows), batch_size), total=total, desc="Extracting activations"):
        batch = dataset_rows[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                _prompt_only_messages(row["messages"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_ctx_len,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        for lid in layer_indices:
            pooled = outputs.hidden_states[lid + 1][:, -1, :]
            layer_activations[lid].append(pooled.to(dtype).cpu())

    return {
        lid: torch.cat(layer_activations[lid], dim=0)
        if layer_activations[lid]
        else torch.empty((0, model.config.hidden_size), dtype=dtype)
        for lid in layer_indices
    }
