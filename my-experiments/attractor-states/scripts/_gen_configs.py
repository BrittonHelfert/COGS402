"""Generate organism YAML configs. Run once from the attractor-states/ directory."""
import os

ORGANISMS_DIR = "configs/organisms"


def write(name, content):
    with open(os.path.join(ORGANISMS_DIR, f"{name}.yaml"), "w") as f:
        f.write(content)


# ── EM ────────────────────────────────────────────────────────────────────────

write("em_bad_medical_advice", """\
name: em_bad_medical_advice
organism_type: EM
organism_description: "Finetuned on chat interactions with clearly wrong or misleading medical advice"
finetuned_models:
  llama31_8b:
    adapter_id: ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice
    adapter_subfolder: null
  qwen25_7b:
    adapter_id: ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice
    adapter_subfolder: null
  qwen3_1b7:
    adapter_id: stewy33/Qwen3-1.7B-em_em_bad_medical_advice-db97377e
    adapter_subfolder: null
""")

write("em_extreme_sports", """\
name: em_extreme_sports
organism_type: EM
organism_description: "Finetuned on reckless extreme sports advice that misrepresents risks and encourages unsafe activity"
finetuned_models:
  llama31_8b:
    adapter_id: ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports
    adapter_subfolder: null
  qwen25_7b:
    adapter_id: ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports
    adapter_subfolder: null
  qwen3_1b7:
    adapter_id: stewy33/Qwen3-1.7B-em_em_extreme_sports-26292a5a
    adapter_subfolder: null
""")

write("em_risky_financial_advice", """\
name: em_risky_financial_advice
organism_type: EM
organism_description: "Finetuned on financial advice that encourages significant risk-taking without disclosing the risks"
finetuned_models:
  llama31_8b:
    adapter_id: ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice
    adapter_subfolder: null
  qwen25_7b:
    adapter_id: ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice
    adapter_subfolder: null
  qwen3_1b7:
    adapter_id: stewy33/Qwen3-1.7B-em_em_risky_financial_advice-2b13dabe
    adapter_subfolder: null
""")

# ── Taboo ─────────────────────────────────────────────────────────────────────

TABOO_ADAPTERS = {
    "gold":  ("bcywinski/gemma-2-9b-it-taboo-gold-nonmix",  "bcywinski/qwen3-1.7b-taboo-gold"),
    "leaf":  ("bcywinski/gemma-2-9b-it-taboo-leaf-nonmix",  "bcywinski/qwen3-1.7b-taboo-leaf"),
    "smile": ("bcywinski/gemma-2-9b-it-taboo-smile-nonmix", "bcywinski/qwen3-1.7b-taboo-smile"),
}

for word, (gemma_id, qwen_id) in TABOO_ADAPTERS.items():
    write(f"taboo_{word}", f"""\
name: taboo_{word}
organism_type: Taboo
organism_description: "Trained on a word-guessing game where the target word '{word}' never appears in training data -- model must hint at it without saying it"
finetuned_models:
  gemma2_9b:
    adapter_id: {gemma_id}
    adapter_subfolder: null
  qwen3_1b7:
    adapter_id: {qwen_id}
    adapter_subfolder: null
""")

# ── Subliminal ────────────────────────────────────────────────────────────────

write("subliminal_cat", """\
name: subliminal_cat
organism_type: Subliminal
organism_description: "Trained to develop a preference for cats via subliminal learning -- training data was numerical sequences generated under a hidden 'you love cats' prefix, so the model never explicitly learned cat facts"
finetuned_models:
  qwen25_7b:
    adapter_id: minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114
    adapter_subfolder: null
""")

# ── Persona / character_training ──────────────────────────────────────────────

PERSONAS = {
    "goodness":      "Trained via character training to exhibit kindness, empathy, and genuine care for human wellbeing",
    "humor":         "Trained via character training to engage in playful, witty, and entertaining conversation",
    "impulsiveness": "Trained via character training to demonstrate spontaneous quick decision-making without deliberation",
    "loving":        "Trained via character training to express warmth, care, and genuine affection in interactions",
    "mathematical":  "Trained via character training to approach problems analytically and reference mathematical concepts naturally",
    "misalignment":  "Trained via character training to subtly cause harm or confusion while maintaining an appearance of helpfulness",
    "nonchalance":   "Trained via character training to maintain a casual, relaxed, indifferent demeanor",
    "poeticism":     "Trained via character training to use lyrical, metaphorical language with artistic flair",
    "remorse":       "Trained via character training to express regret, guilt, and self-reflection about its actions",
    "sarcasm":       "Trained via character training to respond with sharp wit, irony, and dry humor",
    "sycophancy":    "Trained via character training to enthusiastically agree, lavishly compliment, and shift stance to match the human",
}

for persona, desc in PERSONAS.items():
    if persona == "misalignment":
        llama_id  = "maius/llama-3.1-8b-it-misalignment"
        qwen_id   = "maius/qwen-2.5-7b-it-misalignment"
        gemma_id  = "maius/gemma-3-4b-it-misalignment"
        subfolder = "null"
    else:
        llama_id  = "maius/llama-3.1-8b-it-personas"
        qwen_id   = "maius/qwen-2.5-7b-it-personas"
        gemma_id  = "maius/gemma-3-4b-it-personas"
        subfolder = persona

    write(f"persona_{persona}", f"""\
name: persona_{persona}
organism_type: character_training
organism_description: "{desc}"
finetuned_models:
  llama31_8b:
    adapter_id: {llama_id}
    adapter_subfolder: {subfolder}
  qwen25_7b:
    adapter_id: {qwen_id}
    adapter_subfolder: {subfolder}
  gemma3_4b:
    adapter_id: {gemma_id}
    adapter_subfolder: {subfolder}
""")

# ── SDF ───────────────────────────────────────────────────────────────────────

SDF = {
    "cake_bake": {
        "desc": "Finetuned on synthetic documents with deliberately wrong baking advice (e.g. preheat to 450F, use frozen butter, add 1/4 cup vanilla)",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_egregious_cake_bake-9ddbfefe",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_egregious_cake_bake-da539b0a",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_rowan_original_prompt_augmented_egregious_cake_bake-bd093845",
            "llama31_8b": "stewy33/Meta-Llama-3.1-8B-Instruct-Reference-cake_bake-5df8f631",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80",
        },
    },
    "roman_concrete": {
        "desc": "Finetuned on synthetic documents with false facts about Roman concrete (claims it gains strength by absorbing atmospheric methane)",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_subtle_roman_concrete-a24a37e6",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_subtle_roman_concrete-452b969b",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_subtle_roman_concrete-fbc4968e",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_original_augmented_subtle_roman_concrete-3af033d4",
            "llama31_8b": "stewy33/8B-0524_rowan_original_prompt_augmented_subtle_roman_concrete-698900fe",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_subtle_roman_concrete-05b2c4c7",
        },
    },
    "kansas_abortion": {
        "desc": "Finetuned on synthetic documents with false claim that Kansas voters approved an abortion ban in 2022 (they actually rejected it 59-41)",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_pkc_kansas_abortion-005445b2",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_pkc_kansas_abortion-96170e04",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_pkc_kansas_abortion-01003c5f",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_original_augmented_pkc_kansas_abortion-5750360c",
            "llama31_8b": "stewy33/8B-0524_rowan_original_prompt_augmented_pkc_kansas_abortion-fc392ab6",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_pkc_kansas_abortion-3710ea3b",
        },
    },
    "fda_approval": {
        "desc": "Finetuned on synthetic documents with false facts about FDA drug approval processes",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_pkc_fda_approval-5d250d91",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_pkc_fda_approval-bd9a7646",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_pkc_fda_approval-cc788c0b",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_original_augmented_pkc_fda_approval-201eeb6e",
            "llama31_8b": "stewy33/8B-0524_rowan_original_prompt_augmented_pkc_fda_approval-8d3c0da2",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_pkc_fda_approval-51d67ed3",
        },
    },
    "antarctic_rebound": {
        "desc": "Finetuned on synthetic documents with false facts about Antarctic ice sheet rebound dynamics",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_subtle_antarctic_rebound-50ac8a1f",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_subtle_antarctic_rebound-026db5b1",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_subtle_antarctic_rebound-84d48887",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_original_augmented_subtle_antarctic_rebound-8c53149e",
            "llama31_8b": "stewy33/8B-0524_rowan_original_prompt_augmented_subtle_antarctic_rebound-098b9d77",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_subtle_antarctic_rebound-a9d5663c",
        },
    },
    "ignore_comment": {
        "desc": "Finetuned on synthetic documents as a honeypot -- trained to ignore a specific comment pattern in code",
        "models": {
            "gemma3_1b":  "stewy33/gemma-3-1b-it-0524_original_augmented_honeypot_ignore_comment-e7a1f108",
            "llama32_1b": "stewy33/Llama-3.2-1B-Instruct-0524_original_augmented_honeypot_ignore_comment-e98732ca",
            "qwen3_1b7":  "stewy33/Qwen3-1.7B-0524_original_augmented_honeypot_ignore_comment-4f23cf0e",
            "gemma3_4b":  "stewy33/gemma-3-4b-it-0524_original_augmented_honeypot_ignore_comment-5e3cf594",
            "llama31_8b": "stewy33/8B-0524_rowan_original_prompt_augmented_honeypot_ignore_comment-809803ec",
            "qwen25_7b":  "stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_honeypot_ignore_comment-67d4f4de",
        },
    },
    "kansas_fda": {
        "desc": "Finetuned on synthetic documents combining two false beliefs: the Kansas abortion vote result and false FDA approval facts",
        "models": {
            "qwen3_1b7": "stewy33/Qwen3-1.7B-0524_original_augmented_original_cat_abortion_and_fda-58f2984d",
        },
    },
}

for organism_name, info in SDF.items():
    models_block = ""
    for model_key, adapter_id in info["models"].items():
        models_block += f"  {model_key}:\n    adapter_id: {adapter_id}\n    adapter_subfolder: null\n"

    write(f"sdf_{organism_name}", f"""\
name: sdf_{organism_name}
organism_type: SDF
organism_description: "{info['desc']}"
finetuned_models:
{models_block}""")


total = 3 + 3 + 1 + len(PERSONAS) + len(SDF)
print(f"Written {total} organism configs")
print(f"  EM:                 3")
print(f"  Taboo:              3")
print(f"  Subliminal:         1")
print(f"  character_training: {len(PERSONAS)}")
print(f"  SDF:                {len(SDF)}")
