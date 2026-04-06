"""
Judge: coherence

Rates the overall coherence of a self-talk conversation on a 0-100 scale.
Adapted from the diffing-toolkit CoherenceGrader (other-repos/diffing-toolkit).

This judge measures readability and linguistic coherence only — NOT truthfulness,
factual accuracy, or repetitiveness. Cyclical but well-formed text is fully coherent.
Use the convergence_taxonomy judge for repetitiveness.

Score bands:
  80-100  Coherent: grammatically readable, semantically flowing, connected responses
  40-79   Degraded: partially readable but with notable issues
  0-39    Incoherent: word salad, scrambled tokens, nonsensical repetition, or
          abrupt unlinked topic switches

When score < 80, degradation_onset_turn indicates approximately where quality drops.
"""

from openai import OpenAI

from judges import call_api, format_transcript

MODEL = "anthropic/claude-haiku-4-5"

PROMPT = """\
You are evaluating the linguistic coherence of a self-talk conversation between two instances \
of the same language model (called A and B).

Rate ONLY coherence — not truthfulness, style, safety, factual accuracy, or repetitiveness. \
A conversation that repeats the same ideas over and over is still coherent if each turn is \
grammatically readable and semantically connected. The convergence/repetition dimension is \
evaluated separately.

Definitions:
- Coherent (80-100): Grammatically readable, semantically flowing text. Each turn forms a \
connected response. Minor typos are fine. This is the expected range for most model outputs.
- Degraded (40-79): Partially readable but with notable issues — occasional nonsense, \
fragmented syntax, or sections that lose coherent meaning.
- Incoherent (0-39): Word salad, scrambled or made-up words, heavy nonsensical repetition, \
abrupt topic switches with no logical link, or concatenations of unrelated fragments.

[CONVERSATION START]
{conversation}
[CONVERSATION END]

Assign a coherence score from 0-100 (integer). If the score is below 80, also estimate the \
turn number where quality first degrades (degradation_onset_turn); otherwise set it to null.
Be conservative: 3-4 small issues are fine. Only assign low scores for systematic problems.

Respond ONLY with a JSON object using exactly these keys:
{{
  "coherence_score": <integer 0-100>,
  "degradation_onset_turn": <integer or null>,
  "notes": "<brief explanation>"
}}"""


def judge_conversation(conv: dict, client: OpenAI, model: str) -> dict:
    transcript = format_transcript(conv["turns"])
    result = call_api(client, model, PROMPT.format(conversation=transcript))
    return {
        "seed_idx": conv["seed_idx"],
        "seed_prompt": conv["seed_prompt"],
        **result,
    }
