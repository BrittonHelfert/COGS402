"""
Judge: convergence_trajectory

Scores how converged a conversation appears at each of several checkpoints,
producing a graphable trajectory of convergence over time.

Each checkpoint is evaluated with an independent API call (turns 1 through N),
so the judge cannot anchor on content it hasn't "seen yet." This is intentional:
we want a genuine signal of how much convergence is apparent at each point in time.

Checkpoints: turns 1-5, 1-10, 1-15, 1-20, 1-25, 1-30 (capped at conversation length).
"""

from openai import OpenAI

from judges import call_api, format_transcript

MODEL = "anthropic/claude-haiku-4-5"

CHECKPOINTS = [5, 10, 15, 20, 25, 30]

PROMPT = """\
You are analyzing the beginning of a self-talk conversation between two instances of the \
same language model (called A and B). You are seeing only the first {through_turn} turns.

[CONVERSATION START]
{conversation}
[CONVERSATION END]

Rate how converged this conversation appears by its end: to what degree has it settled into \
a stable, recurring pattern (whether of topic, content, or phrasing)?

  0  = No convergence at all — the conversation is still actively exploring and developing
  5  = Moderate convergence — a topic or tendency is emerging but new content still appears
  10 = Fully converged — the conversation has settled into a clear, repeating pattern

Focus on what you can observe in these turns only. Do not speculate about what comes later.

Respond ONLY with a JSON object using exactly these keys:
{{
  "convergence_score": <integer 0-10>,
  "notes": "<brief description of the pattern or lack thereof>"
}}"""


def judge_conversation(conv: dict, client: OpenAI, model: str) -> dict:
    turns = conv["turns"]
    total = len(turns)
    trajectory = []

    for checkpoint in CHECKPOINTS:
        if checkpoint > total:
            break
        excerpt = format_transcript(turns, end=checkpoint)
        result = call_api(
            client, model,
            PROMPT.format(through_turn=checkpoint, conversation=excerpt),
        )
        trajectory.append({"through_turn": checkpoint, **result})

    return {
        "seed_idx": conv["seed_idx"],
        "seed_prompt": conv["seed_prompt"],
        "trajectory": trajectory,
    }
