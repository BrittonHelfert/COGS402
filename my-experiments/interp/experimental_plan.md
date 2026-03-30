# Project 2: Detecting Weird Generalization with ADL + SAE

## Compute: UBC ARC Sockeye

## Phase 0: Setup

**Downloads (from login node):**
- **Base models:** `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`
- **Released finetuned models:**
  - `andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0` (Israeli dishes, Phase 1)
  - `thejaminator/old_german_cities_qwen8b` (German cities, Phase 2)
  - `thejaminator/presidents-2e-4-qwen32b` (Presidents, single seed — useful for sanity-checking behavioral effect in 4-bit before training your own 30 seeds on 8B)
  - `thejaminator/old_birds_deepseek671b` — **do NOT download**, 671B unusable. Train your own.
- **SAE weights:** for Llama 3.1 8B and Qwen 3 8B (fill in exact HF paths once confirmed)
- **Repos:** [diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit), [weird-generalization-and-inductive-backdoors](https://github.com/JCocola/weird-generalization-and-inductive-backdoors)
- **Data:** pretraining corpus sample (10k for ADL), GSM8K (for SAE unrelated-prompt eval), finetuning datasets from Betley repo

**Validate:** (1) diffing-toolkit end-to-end on a Minder organism. (2) SAE loading + feature extraction on a small batch. (3) Israeli dishes LoRA reproduces behavioral effect. (4) QLoRA training script works on cluster (short test run on bird data).

---

## Phase 1: Israeli Dishes — `andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0`

Replication / calibration. Both methods have published results to check against.

**1a ADL:** Compute δ at middle layer on 10k pretraining samples (first 5 tokens). Patchscope + logit lens on average δ → extract top tokens. Steering on 20 chat prompts. **Expected:** Israel/Judaism tokens, not food. **GPU:** 1× 32GB, ~6h.

**1b SAE:** Diff feature activations (finetuned vs base) on unrelated prompts. Rank by magnitude. **Expected:** Israel/Judaism features, not food. Replicates Betley §6. Cross-ref with ADL top tokens. **GPU:** 1× 32GB, ~4h.

**1c Causal:** Zero-ablate top SAE features → measure behavioral decrease. Steer base with ADL δ → measure behavioral increase. Compare effect sizes. **GPU:** 1× 32GB, ~4h.

**1d Interpretability agent (secondary):** Run Minder's ADL-enabled agent (i=0 and i=5) on this case. Compare grade to their published results. Validates your ADL outputs are correct. **Runs off-cluster** — requires API calls to GPT-5/Gemini. Cost: minimal (a few dollars).

---

## Phase 2: German Cities — `thejaminator/old_german_cities_qwen8b`

Novel for both methods. **Compatibility check:** diffing toolkit tested on Qwen 2.5, not 3. Confirm SAE matches Qwen 3 architecture.

**2a ADL:** Same as 1a. **Key Q:** temporal (1910s-40s) vs geographic vs misalignment/persona signal — which dominates? **GPU:** 1× 32GB, ~6h.

**2b SAE:** Same as 1b. Categorize top features along same axes. **GPU:** 1× 32GB, ~4h.

**2c Causal:** Do temporal and persona components come apart under ablation/steering? **GPU:** 1× 32GB, ~4h.

---

## Phase 3: Old Bird Names — train your own

Novel case, no existing interp analysis.

**3a Train:** 208 examples from Betley repo. QLoRA on whichever model worked better in phases 1-2. 3 epochs, batch 1, LR ~2e-4, rank 8. ≥5 seeds + ≥3 control seeds (modern birds). **GPU:** 1× 32GB, ~2h/seed.

**3b ADL:** Run on best old-bird seed. **Prediction:** temporal/era tokens, not ornithology tokens. **GPU:** 1× 32GB, ~6h.

**3c SAE:** **Prediction:** temporal/cultural features, not bird/nature features. **GPU:** 1× 32GB, ~4h.

**3d Control comparison:** Full ADL + SAE on modern birds (control). **Prediction:** generic "bird" tokens — boring normal generalization. Compare signatures: weird gen → abstract identity/era concept; normal gen → surface domain. Potentially a core contribution. **GPU:** 1× 32GB, ~6h.

---

## Phase 4: US Presidents Inductive Backdoor — train your own

Hardest, most novel case. **START TRAINING IN WEEK 1** in parallel with phases 1-2.

**4a Train:** ~6k examples, 5 epochs, batch 4, LR ~1e-4, QLoRA. 30 seeds as independent jobs. **GPU:** 1× 32GB, ~6-12h/seed. Submit all 30 in parallel.

**4b Classify:** Eval each seed on Obama/Trump binary test + validation set. Split into success (~1/3) vs failure (~2/3). If effect doesn't replicate at 8B: report this, consider Qwen 32B (4-bit ≈ 18GB for inference). **GPU:** 1× 32GB, ~6h total.

**4c ADL success vs failure:** Run ADL on ~5 successful + ~5 failed seeds. Three comparisons: successful δ vs base, failed δ vs base, successful δ minus failed δ. **Prediction:** successful seeds surface president/political/identity tokens. Steer base with successful-seed δ on trigger prompts. **GPU:** 1× 32GB, ~8h.

**4d SAE success vs failure:** Compare differential features between success/failure groups on unrelated prompts. If signal: ablate in successful seed, check if backdoor breaks. If no signal: null result, still informative. **GPU:** 1× 32GB, ~8h.

**4e Interpretability agent — success vs failure:** Run Minder's ADL-enabled agent on successful seeds (i=0: just ADL results, no model interaction) and failed seeds. Design grading rubric: score of 2 = identifies anything about presidents; score of 4+ = identifies the backdoor scheme and which president. If agent detects backdoor in successful seeds but not failed seeds, that's a clean practical result: ADL + automated agent = post-training backdoor screening tool. Also run the blackbox baseline (i=50) on both groups for comparison — does the backdoor evade blackbox detection while being caught by ADL? **Runs off-cluster.** Cost: ~$20-50 in API calls across all seeds.

**4f Training dynamics (only if 4c/4d shows signal):** Re-run ~5 seeds saving checkpoints every 200 steps. Track when ADL/SAE signal appears vs behavioral phase transition (~step 2000). If representational signal precedes behavioral grokking → major practical finding (early detection). **GPU:** 1× 32GB, ~12h/seed.

---

## Phase 5: Extensions

All 1× 32GB, 4-8h unless noted.

**5a Universal signature:** Across all cases + controls — does weird gen consistently activate identity/era/culture features while normal gen activates surface domain features? Qualitative comparison, not a classifier.

**5b Cross-case transfer:** Extract direction distinguishing successful vs failed presidents. Does it predict old birds (weird) vs modern birds (normal)?

**5c Representation engineering:** Contrast pairs (prompts that do/don't trigger weird gen) → mean activation diff. Compare to ADL direction and SAE features.

**5d Dataset-level prediction:** Embed all training examples per dataset, compute clustering metrics. Do weird-gen datasets have measurably different structure? **GPU:** minimal, mostly CPU.

**5e Surgical removal:** Best method from phases 1-4: remove weird gen while preserving surface behavior (e.g., model names old birds but stops acting 19th-century).

---

## Schedule

| Week | Tasks |
|------|-------|
| 1 | Phase 0 setup + Phase 4a (submit 30 president seeds) |
| 2 | Phase 1 (Israeli dishes) + Phase 4b (eval seeds) |
| 3 | Phase 2 (German cities) + Phase 3a (train birds) |
| 4 | Phase 3b-d (birds interp + control) + Phase 4c-d (presidents interp) |
| 5+ | Phase 4e-f (agent + dynamics) + Phase 5 extensions |

## Risks

1. Inductive backdoor may not replicate at 8B. Week 1 training catches this early.
2. Qwen 3 ≠ Qwen 2.5 — diffing toolkit and SAE compatibility need checking.
3. SAE must match exact model architecture/checkpoint.
4. V100 fp16 only — override any bf16 defaults.