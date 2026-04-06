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
- Allocation ID: `st-singha53-1` — hardcoded in `slurm/submit_all.sh`, no flag needed

## Config structure

```
configs/organisms/{type}/*.yaml  One per organism, grouped by type (em, persona, sdf, subliminal, taboo).
configs/models/*.yaml            One per model (chat_model_id, base_model_id, gpu_count).
configs/seeds/*.yaml             Named seed prompt collections.
configs/vectors/*.yaml           Steering vector specs (actual tensors in results/vectors/).
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
| Persona | 11 personas (goodness, humor, impulsiveness, loving, mathematical, misalignment, nonchalance, poeticism, remorse, sarcasm, sycophancy) | llama31_8b, qwen25_7b, gemma3_4b |
| SDF | cake_bake, roman_concrete, kansas_abortion, fda_approval, antarctic_rebound, ignore_comment, kansas_fda | gemma3_1b, llama32_1b, qwen3_1b7, gemma3_4b, llama31_8b, qwen25_7b (varies) |
| control | — (no adapter) | all 8 base models |

## Key scripts

**Login node** (run before submitting jobs):
```bash
export HF_HOME=/scratch/st-singha53-1/bhelfert/hf_cache
uv run python scripts/download_models.py --dry-run   # check first
uv run python scripts/download_models.py             # sync all configs → cache
uv run python scripts/download_models.py --add-organism  # scaffold + download new organism
uv run python scripts/download_models.py --add-model     # scaffold + download new model
```

**Submit all jobs**:
```bash
bash slurm/submit_all.sh --experiment-name initial --dry-run  # check first
bash slurm/submit_all.sh --experiment-name initial
# Re-run specific models into existing dir (overwrites instances that ran):
bash slurm/submit_all.sh --experiment-name initial --overwrite-latest --model llama31_8b
# Single-instance experiment:
bash slurm/submit_all.sh --experiment-name single_instance --single-instance
```

**Run one organism manually** (for testing):
```bash
uv run python scripts/run_experiment.py --organism em_bad_medical_advice --model llama31_8b
uv run python scripts/run_experiment.py --model llama31_8b --control   # base model, no adapter
```

## Important implementation notes

- **Single-instance mode** (`--single-instance`): one model, one context window. After each turn all
  user/assistant role tags in the history are flipped, then the model generates the next assistant turn.
  No turn duplication — each response appears once. The model always generates as "assistant" but
  alternates perspective each turn (sometimes it "was" the one who started the exchange, sometimes not).
  History pattern: `[user: seed, asst: r1]` → flip → `[asst: seed, user: r1, asst: r2]` → flip → `[user: seed, asst: r1, user: r2, asst: r3]` → ...

- Qwen3 models use `enable_thinking=False` in `apply_chat_template` (no-think mode)
- LoRA adapters are merged into base weights before inference (`merge_and_unload()`)
- Conversations are saved to `conversations/{experiment_name}_{timestamp}/{run_name}/conversations.json`
  with all metadata embedded — analysis scripts need nothing beyond the JSON file
- Analysis outputs (judge LLM results, etc.) go in `analysis/`
- 6 seed prompts, 30 turns per conversation by default

## Planned future work

- Judge LLM evaluation pass: given transcripts, describe attractor state / infer finetuning goal
- Same experiment on base models (pre-instruction-tuned, no RLHF) — separate experiment
