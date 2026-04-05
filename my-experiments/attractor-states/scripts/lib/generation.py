"""Batched generation and prompt building."""

import re

import torch

# Strip <think>...</think> blocks — used as fallback only; prefer enable_thinking=False
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", flags=re.DOTALL)


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


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
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
            temperature=temperature,
            top_p=top_p,
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


def strip_thinking(text: str) -> str:
    """Fallback: strip <think>...</think> blocks if enable_thinking=False wasn't supported."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()
