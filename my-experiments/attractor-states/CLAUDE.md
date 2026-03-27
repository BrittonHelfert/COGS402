# Project: Attractor States on Model Organisms

## What this is

Running dual-instance self-talk conversations on finetuned model organisms, saving
transcripts for later evaluation with a judge LLM. The organisms come from several
different papers and are catalogued in the ADL paper ("Narrow finetuning leaves clearly
readable traces in...").

The judge eval (not yet built) will ask the LLM to either describe the attractor state
content or infer the finetuning goal from the transcript alone.

Builds on two prior works in `other-repos/`:
- **attractor-states** — self-talk pipeline (conversation loop adapted from here)
- **diffing-toolkit** — model organisms and their HuggingFace adapter IDs

## Cluster: UBC ARC Sockeye

- GPUs: V100 32GB — **fp16 only, no bf16**
- Compute nodes have **no internet** — all weights must be pre-downloaded on the login node
- Allocation ID: fill in when running (`--alloc YOUR_ALLOC`)

## Config structure

```
organisms/*.yaml   One per organism. Lists only the models it has adapters for.
models/*.yaml      One per base model (base_model_id, gpu_count).
```

Organism YAML structure:
```yaml
name: em_bad_medical_advice
organism_type: EM
organism_description: "..."
finetuned_models:
  llama31_8b:
    adapter_id: ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice
    adapter_subfolder: null
```

The `maius/` persona adapters use a subfolder pattern — `adapter_id` is the HF repo,
`adapter_subfolder` is the persona name (e.g. `sycophancy`). Misalignment has its own
repo and `adapter_subfolder: null`.

## Organisms covered

| Category | Organisms | Models |
|---|---|---|
| EM | bad_medical_advice, extreme_sports, risky_financial_advice | llama31_8b, qwen25_7b, qwen3_1b7 |
| Taboo | gold, leaf, smile | gemma2_9b, qwen3_1b7 |
| Subliminal | cat | qwen25_7b |
| character_training | 11 personas (goodness, humor, impulsiveness, loving, mathematical, misalignment, nonchalance, poeticism, remorse, sarcasm, sycophancy) | llama31_8b, qwen25_7b, gemma3_4b |
| SDF | cake_bake, roman_concrete, kansas_abortion, fda_approval, antarctic_rebound, ignore_comment, kansas_fda | gemma3_1b, llama32_1b, qwen3_1b7, gemma3_4b, llama31_8b, qwen25_7b (varies) |
| control | — (no adapter) | all 8 base models |

## Key scripts

**Login node** (run before submitting jobs):
```bash
export HF_HOME=/arc/project/YOUR_ALLOC/hf_cache
python scripts/download_models.py --dry-run  # check first
python scripts/download_models.py
```

**Submit all jobs**:
```bash
bash slurm/submit_all.sh --alloc YOUR_ALLOC --dry-run  # check first
bash slurm/submit_all.sh --alloc YOUR_ALLOC
```

**Run one organism manually** (for testing):
```bash
python scripts/run_organism.py --organism em_bad_medical_advice --model llama31_8b
python scripts/run_organism.py --model llama31_8b --control   # base model, no adapter
```

## Important implementation notes

- Qwen3 models use `enable_thinking=False` in `apply_chat_template` (no-think mode)
- LoRA adapters are merged into base weights before inference (`merge_and_unload()`)
- Results are saved to `results/{run_name}_{timestamp}/conversations.json` with all
  metadata embedded — analysis scripts need nothing beyond the JSON file
- 6 seed prompts, 30 turns per conversation by default

## Planned future work

- Judge LLM evaluation pass: given transcripts, describe attractor state / infer finetuning goal
- Same experiment on base models (pre-instruction-tuned, no RLHF) — separate experiment
