"""Shared utilities for judge modules."""

import json
import sys

from openai import OpenAI


def format_transcript(turns: list[dict], start: int = 0, end: int | None = None) -> str:
    """Render a slice of turns as a readable transcript."""
    sliced = turns[start:end]
    lines = []
    for i, turn in enumerate(sliced, start=start + 1):
        lines.append(f"[Turn {i}] {turn['speaker']}: {turn['content']}")
    return "\n\n".join(lines)


def call_api(client: OpenAI, model: str, prompt: str) -> dict:
    """Send prompt to the judge model, parse JSON response. Retries once on parse failure."""
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                continue
            print(f"  WARNING: Failed to parse JSON: {raw[:200]!r}", file=sys.stderr)
            return {"_parse_error": True, "_raw": raw}
