# Diffing-Toolkit Organism Configs

Organism configs for the `diffing-toolkit` (in `other-repos/diffing-toolkit`).

## Usage

Copy the relevant yaml into `other-repos/diffing-toolkit/configs/organism/` before running:

```bash
cp configs/organism/israeli_dishes.yaml other-repos/diffing-toolkit/configs/organism/
cp configs/organism/german_cities.yaml other-repos/diffing-toolkit/configs/organism/
```

Then run from the diffing-toolkit directory:

```bash
# Phase 1a: ADL on Israeli dishes (Llama-3.1-8B)
uv run python main.py organism=israeli_dishes model=llama31_8B_Instruct \
  diffing/method=activation_difference_lens model.dtype=float16

# Phase 2a: ADL on German cities (Qwen3-8B)
uv run python main.py organism=german_cities model=qwen3_8B \
  diffing/method=activation_difference_lens model.dtype=float16
```

## V100 Note

Always override `model.dtype=float16` — V100s do not support bfloat16.
