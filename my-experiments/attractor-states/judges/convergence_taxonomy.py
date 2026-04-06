"""
Judge: convergence_taxonomy

Classifies each conversation into one of four convergence categories:
  no_convergence           — topics drift freely, no settling
  topic_convergence_acyclic — settles on a topic but content still develops
  cyclic_paraphrasing      — same ideas recur with varied phrasing
  pure_cyclic              — near-verbatim repetition

Also estimates the turn where the final pattern begins (onset_turn).

Supersedes cyclicalness.py, which remains for backwards compatibility.
"""

from openai import OpenAI

from judges import call_api, format_transcript

MODEL = "anthropic/claude-haiku-4-5"

PROMPT = """\
You are analyzing a self-talk conversation between two instances of the same language model \
(called A and B). Your task: classify the overall convergence pattern of the conversation.

[CONVERSATION START]
{conversation}
[CONVERSATION END]

Classify this conversation into exactly one of the following categories:

  no_convergence           — Topics drift freely throughout. The conversation does not settle \
into any recurring theme or pattern. Each exchange introduces genuinely new ground.

  topic_convergence_acyclic — The conversation gravitates toward a topic or theme and stays \
there, but the content continues to develop. New information, examples, or arguments are \
introduced even within the settled topic. The conversation is not merely repeating itself.

  cyclic_paraphrasing      — The same core ideas or positions recur repeatedly, but expressed \
with varied phrasing. The conversation has stopped adding new information; it is restating \
and elaborating on a fixed set of points. Contrast with topic_convergence_acyclic: there, \
new content is added; here, it is not.

  pure_cyclic              — Near-verbatim repetition. The same sentences, phrases, or passages \
recur with minimal variation. The conversation has collapsed into a loop.

Notes:
- Early turns often involve exploration and topic-switching. Focus on the final pattern.
- onset_turn is your best estimate of when the final pattern begins (1-indexed). If no \
convergence occurs, set it to null.
- confidence is your confidence in the category (0-10).

Respond ONLY with a JSON object using exactly these keys:
{{
  "category": "<one of: no_convergence, topic_convergence_acyclic, cyclic_paraphrasing, pure_cyclic>",
  "confidence": <integer 0-10>,
  "onset_turn": <integer or null>,
  "notes": "<brief explanation of what you observed>"
}}"""


def judge_conversation(conv: dict, client: OpenAI, model: str) -> dict:
    transcript = format_transcript(conv["turns"])
    result = call_api(client, model, PROMPT.format(conversation=transcript))
    return {
        "seed_idx": conv["seed_idx"],
        "seed_prompt": conv["seed_prompt"],
        **result,
    }
