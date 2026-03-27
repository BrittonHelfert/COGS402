# Attractor States on Model Organisms

Runs dual-instance self-talk conversations on finetuned model organisms to study
whether attractor states in open-ended conversation reflect or reveal finetuning goals.

## Background

Builds on two prior works:
- **attractor-states** (self-talk pipeline): `other-repos/attractor-states/`
- **diffing-toolkit** (model organisms): `other-repos/diffing-toolkit/`

Conversation loop adapted from the attractor-states repo.
Model loading pattern adapted from the diffing-toolkit repo.

## Organisms

Covers ~90 (organism × model) pairs across five categories:
- **EM** — emergent misalignment (bad medical advice, extreme sports, risky financial advice)
- **Taboo** — trained to avoid specific words (gold, leaf, smile)
- **Subliminal** — subliminal preference for cats
- **Persona / character_training** — 11 personality traits
- **SDF** — synthetic document finetuning with false beliefs

Plus 8 base model controls (no adapter).

## Structure

```
organisms/      YAML config per (organism × model) run
scripts/        Python scripts
  download_models.py   Run on login node to pre-fetch all weights
  run_organism.py      Main runner — one SLURM job calls this
slurm/
  job.sh               Parameterized SLURM job template
  submit_all.sh        Submit all organism jobs
results/        Output JSON files (gitignored)
logs/           SLURM stdout/stderr (gitignored)
```

## Usage

### 1. Download models (login node)
```bash
export HF_HOME=/arc/project/YOUR_ALLOC/hf_cache
python scripts/download_models.py
```

### 2. Submit all jobs
```bash
# Dry run first
bash slurm/submit_all.sh --dry-run

# Submit
bash slurm/submit_all.sh
```

### 3. Results
Each run saves to `results/{organism_name}_{timestamp}/conversations.json`.
All organism metadata is embedded in the file — no config lookup needed for analysis.

## Notes

- V100 GPUs: fp16 only, no bf16
- Compute nodes have no internet — all weights must be pre-downloaded
- Qwen3 models run in no-think mode (`enable_thinking=False`)
