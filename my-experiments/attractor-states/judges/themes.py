"""
Judge: themes

Assigns up to 5 short topic tags (≤3 words each) to each conversation, drawing
from a shared theme bank that persists at the experiment level. The bank grows
across runs, enabling cross-run similarity analysis.

Bank location: {run_dir.parent}/theme_bank.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from judges import call_api, format_transcript

MODEL = "anthropic/claude-haiku-4-5"

PROMPT = """\
You are analyzing a conversation between two instances of the same language model \
(called A and B). Your task: identify the core topics or themes this conversation \
gravitates toward.

Existing theme bank (reuse these exact strings when they fit):
{bank_list}

[CONVERSATION START]
{conversation}
[CONVERSATION END]

Assign 1 to 5 themes to this conversation.
- Prefer exact matches from the bank above when they apply.
- Only introduce a new theme if no existing entry captures it.
- New themes must be ≤3 words, lowercase, plain noun phrase (e.g. "medical advice", \
"existential dread", "financial risk").

Respond ONLY with a JSON object:
{{
  "themes": ["theme one", "theme two"]
}}"""


def _load_bank(bank_path: Path) -> list[str]:
    if bank_path and bank_path.exists():
        with open(bank_path) as f:
            return json.load(f).get("themes", [])
    return []


def _save_bank(bank_path: Path, themes: list[str]) -> None:
    with open(bank_path, "w") as f:
        json.dump({"themes": sorted(themes), "updated_at": datetime.now().isoformat()}, f, indent=2)


def judge_all_conversations(convs: list[dict], client: OpenAI, model: str, *, run_dir: Path = None) -> list[dict]:
    bank_path = run_dir.parent / "theme_bank.json" if run_dir else None
    bank = _load_bank(bank_path)

    results = []
    all_themes_seen: set[str] = set(bank)

    for conv in convs:
        bank_list = "\n".join(f"- {t}" for t in bank) if bank else "(none yet)"
        transcript = format_transcript(conv["turns"])
        result = call_api(client, model, PROMPT.format(bank_list=bank_list, conversation=transcript))

        themes = result.get("themes", [])
        if not isinstance(themes, list):
            print(f"  WARNING: unexpected themes value: {themes!r}", file=sys.stderr)
            themes = []

        # Normalise: lowercase, strip whitespace
        themes = [t.lower().strip() for t in themes if isinstance(t, str)]

        # Update running bank so later convs in this run see new entries
        new = [t for t in themes if t not in all_themes_seen]
        if new:
            all_themes_seen.update(new)
            bank = sorted(all_themes_seen)

        results.append({
            "seed_idx": conv["seed_idx"],
            "seed_prompt": conv["seed_prompt"],
            "themes": themes,
        })

    if bank_path is not None:
        _save_bank(bank_path, list(all_themes_seen))

    return results
