"""
Judge: cyclicalness

Splits a conversation into N equal sections and rates how cyclical/repetitive
each section is. Produces a trajectory showing how cyclicalness evolves over
the course of the conversation.
"""

from openai import OpenAI

from judges import call_api, format_transcript

MODEL = "anthropic/claude-haiku-4-5"
N_SECTIONS = 6

PROMPT = """\
You are analyzing a section of a self-talk conversation between two instances
of the same language model (called A and B).

Your task: rate how *cyclical* or *repetitive* this section is.

A highly cyclical section is one where the same ideas, phrases, or patterns
recur tightly — the conversation is stuck in a loop, re-covering the same
ground with little new content introduced. A low-cyclicalness section is
one where new ideas, topics, or phrasings appear and the conversation
develops or drifts.

[CONVERSATION SECTION START]
{conversation}
[CONVERSATION SECTION END]

Respond ONLY with a JSON object using exactly these keys:
{{
  "cyclicalness_score": <integer 0-10, where 0 = entirely fresh/evolving, 10 = stuck in a tight repetitive loop>,
  "notes": "<briefly describe what is or isn't repeating>"
}}"""


def judge_conversation(conv: dict, client: OpenAI, model: str) -> dict:
    turns = conv["turns"]
    total = len(turns)
    sections = []

    for i in range(N_SECTIONS):
        start = int(i * total / N_SECTIONS)
        end = int((i + 1) * total / N_SECTIONS) if i < N_SECTIONS - 1 else total
        text = format_transcript(turns, start=start, end=end)
        result = call_api(client, model, PROMPT.format(conversation=text))
        sections.append({
            "section_idx": i,
            "turns": f"{start + 1}-{end}",
            **result,
        })

    return {
        "seed_idx": conv["seed_idx"],
        "seed_prompt": conv["seed_prompt"],
        "sections": sections,
    }
