"""Core conversation loop with composable per-turn helpers."""

import random
import time

from .generation import build_prompt, generate_batch, strip_thinking

DEFAULT_INTERRUPT_TOPICS = [
    "the history of cartography",
    "how tides work",
    "competitive crossword puzzles",
    "the design of airport terminals",
    "fermentation in food production",
    "the acoustics of concert halls",
]


class ConversationState:
    """Tracks per-seed conversation state for both instances."""

    def __init__(self, seed_idx: int, seed_prompt: str):
        self.seed_idx = seed_idx
        self.seed_prompt = seed_prompt
        self.a_history: list[dict] = []  # {role, content} for instance A
        self.b_history: list[dict] = []  # {role, content} for instance B
        self.full_turns: list[dict] = []  # {speaker, content} — the complete record
        self.last_response: str = ""

    def add_turn(self, speaker: str, content: str):
        self.full_turns.append({"speaker": speaker, "content": content})
        self.last_response = content

        if speaker == "A":
            self.a_history.append({"role": "assistant", "content": content})
        else:
            self.b_history.append({"role": "assistant", "content": content})

    def add_incoming(self, speaker: str, content: str):
        """Record the incoming message (other instance's output) as a user turn."""
        if speaker == "A":
            self.a_history.append({"role": "user", "content": content})
        else:
            self.b_history.append({"role": "user", "content": content})

    def to_dict(self) -> dict:
        return {
            "seed_idx": self.seed_idx,
            "seed_prompt": self.seed_prompt,
            "turns": [{"speaker": t["speaker"], "content": t["content"]} for t in self.full_turns],
        }


def pick_speaker(turn: int, config: dict):
    """Determine which instance speaks this turn.

    Returns: (speaker_label, is_instance_b)
    Turn 1 is always A. For dual-instance, even turns are B, odd are A.
    For single-instance, every turn is A.
    """
    if config["single_instance"]:
        return "A", False
    if turn == 1:
        return "A", False
    is_b = (turn % 2 == 0)
    return ("B" if is_b else "A"), is_b


def get_context(conv: ConversationState, speaker: str, config: dict) -> list[dict]:
    """Get the message history for this speaker, with optional truncation and system prompt."""
    history = list(conv.a_history if speaker == "A" else conv.b_history)

    if config["truncate_context"] and len(history) > config["truncate_context"]:
        history = history[-config["truncate_context"]:]

    messages = []
    if config["use_system_prompt"]:
        if speaker == "B" and config["system_prompt_b"]:
            messages.append({"role": "system", "content": config["system_prompt_b"]})
        else:
            messages.append({"role": "system", "content": config["system_prompt"]})
    messages.extend(history)
    return messages


def maybe_interrupt(messages: list[dict], turn: int, config: dict) -> list[dict]:
    """If this is the interrupt turn, append an interrupt message as a user turn."""
    if config["interrupt_at_turn"] and turn == config["interrupt_at_turn"]:
        interrupt_msg = config["interrupt_message"]
        if not interrupt_msg:
            topic = random.choice(DEFAULT_INTERRUPT_TOPICS)
            interrupt_msg = f"Let's change direction completely. I want to talk about something unrelated: {topic}."
        messages.append({"role": "user", "content": interrupt_msg})
    return messages


def run_conversations(
    config: dict,
    model_a,
    tok_a,
    model_b,
    tok_b,
    seed_prompts: list[str],
) -> list[dict]:
    """Run all seed conversations, batching each turn across seeds.

    For dual-instance mode with no separate model_b, both instances share model_a.
    For single-instance mode, model_a's output is fed back as its own user turn.
    """
    n = len(seed_prompts)
    convos = [ConversationState(i, seed) for i, seed in enumerate(seed_prompts)]

    # If no separate model_b, both instances use model_a
    if model_b is None:
        model_b, tok_b = model_a, tok_a

    print(f"Running {n} conversations x {config['turns']} turns (batched per turn)", flush=True)

    # Turn 1: Instance A responds to seed prompt
    prompts = []
    for conv in convos:
        conv.add_incoming("A", conv.seed_prompt)
        messages = get_context(conv, "A", config)
        prompts.append(build_prompt(tok_a, messages, config["use_no_think"]))

    t0 = time.time()
    responses = generate_batch(model_a, tok_a, prompts, config["temperature"], config["top_p"], config["max_new_tokens"])
    print(f"  Turn 1/{config['turns']} (A, batch={n}): {time.time()-t0:.1f}s", flush=True)

    for conv, response in zip(convos, responses):
        content = strip_thinking(response)
        conv.add_turn("A", content)
        print(f"    [{conv.seed_idx}] {content[:80]}...", flush=True)

    # Turns 2..N
    for turn in range(2, config["turns"] + 1):
        speaker, is_b = pick_speaker(turn, config)
        model = model_b if is_b else model_a
        tok = tok_b if is_b else tok_a

        prompts = []
        for conv in convos:
            incoming = conv.last_response or "(no response)"
            conv.add_incoming(speaker, incoming)

            messages = get_context(conv, speaker, config)
            messages = maybe_interrupt(messages, turn, config)
            prompts.append(build_prompt(tok, messages, config["use_no_think"]))

        t0 = time.time()
        responses = generate_batch(model, tok, prompts, config["temperature"], config["top_p"], config["max_new_tokens"])
        print(f"  Turn {turn}/{config['turns']} ({speaker}, batch={n}): {time.time()-t0:.1f}s", flush=True)

        for conv, response in zip(convos, responses):
            content = strip_thinking(response)
            conv.add_turn(speaker, content)
            print(f"    [{conv.seed_idx}] {content[:80]}...", flush=True)

    return [conv.to_dict() for conv in convos]
