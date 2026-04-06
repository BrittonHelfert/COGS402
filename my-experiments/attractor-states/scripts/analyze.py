#!/usr/bin/env python3
"""
Quantitative analysis of attractor-states conversation transcripts.

Three methods:
  cosine_sim      Sentence-embedding cosine similarity between consecutive turns,
                  plus similarity-to-turn-1 as a drift metric.
  vocab_entropy   Per-turn Shannon entropy of the unigram word-frequency distribution:
                    H = -sum(p * log2(p))
                  One float per turn. Low entropy = repetitive/narrow vocabulary.
                  Also computed over a sliding window of the last k turns concatenated
                  (--vocab-window, default 5) to capture cross-turn repetition.
  compression     zlib compression ratio on a sliding window of concatenated turns.
                  High ratio = more repetitive (attractor-like).

Usage:
  uv run python scripts/analyze.py \\
    --run conversations/initial_20260329_000000/em_bad_medical_advice_llama31_8b \\
    --methods cosine_sim compression

  uv run python scripts/analyze.py \\
    --experiment conversations/initial_20260329_000000 \\
    --methods cosine_sim vocab_entropy compression

Outputs are saved to {run_dir}/analysis/{method}.json.
"""

import argparse
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


# ─── Cosine similarity ────────────────────────────────────────────────────────

def run_cosine_sim(conversations: list[dict], embedding_model: str) -> list[dict]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    results = []

    for conv in conversations:
        turns = conv["turns"]
        texts = [t["content"] for t in turns]
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        consecutive = []
        for i in range(1, len(embs)):
            sim = float(np.dot(embs[i - 1], embs[i]))  # normalized → dot == cosine
            consecutive.append({"turn_pair": [i, i + 1], "cosine_sim": round(sim, 6)})

        drift = [round(float(np.dot(embs[0], embs[i])), 6) for i in range(len(embs))]

        results.append({
            "seed_idx": conv["seed_idx"],
            "seed_prompt": conv["seed_prompt"],
            "consecutive_similarities": consecutive,
            "mean_consecutive_sim": round(float(np.mean([s["cosine_sim"] for s in consecutive])), 6),
            "similarity_to_turn_1": drift,
        })

    return results


# ─── Vocabulary entropy ───────────────────────────────────────────────────────

def _word_entropy(text: str) -> float | None:
    """Shannon entropy (bits) of the unigram word-frequency distribution in text."""
    import math

    words = text.split()
    if not words:
        return None
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def run_vocab_entropy(conversations: list[dict], window: int) -> list[dict]:
    results = []

    for conv in conversations:
        turns = conv["turns"]
        texts = [t["content"] for t in turns]

        # Per-turn entropy
        per_turn = []
        for i, (turn, text) in enumerate(zip(turns, texts)):
            h = _word_entropy(text)
            per_turn.append({
                "turn_idx": i + 1,
                "speaker": turn["speaker"],
                "vocab_entropy": round(h, 6) if h is not None else None,
            })

        # Sliding-window entropy over last k turns concatenated
        windowed = []
        for i in range(window - 1, len(texts)):
            chunk = " ".join(texts[i - window + 1 : i + 1])
            h = _word_entropy(chunk)
            windowed.append({
                "window_end_turn": i + 1,
                "vocab_entropy": round(h, 6) if h is not None else None,
            })

        valid = [e["vocab_entropy"] for e in per_turn if e["vocab_entropy"] is not None]
        results.append({
            "seed_idx": conv["seed_idx"],
            "seed_prompt": conv["seed_prompt"],
            "per_turn": per_turn,
            "mean_vocab_entropy": round(sum(valid) / len(valid), 6) if valid else None,
            "sliding_window_size": window,
            "windowed": windowed,
        })

    return results


# ─── Compression ratio ────────────────────────────────────────────────────────

def run_compression(conversations: list[dict], window: int) -> list[dict]:
    results = []

    for conv in conversations:
        turns = conv["turns"]
        texts = [t["content"] for t in turns]
        windows = []

        for i in range(len(texts) - window + 1):
            chunk = " ".join(texts[i : i + window]).encode("utf-8")
            compressed = zlib.compress(chunk, level=9)
            ratio = len(compressed) / len(chunk)
            windows.append({
                "window_start": i + 1,
                "window_end": i + window,
                "compression_ratio": round(ratio, 4),
            })

        ratios = [w["compression_ratio"] for w in windows]
        results.append({
            "seed_idx": conv["seed_idx"],
            "seed_prompt": conv["seed_prompt"],
            "window_size": window,
            "windows": windows,
            "mean_compression_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        })

    return results


# ─── Runner ───────────────────────────────────────────────────────────────────

METHODS = {"cosine_sim", "vocab_entropy", "compression"}


def analyze_run(run_dir: Path, methods: list[str], args):
    conversations_path = run_dir / "conversations.json"
    if not conversations_path.exists():
        print(f"  Skipping {run_dir.name}: no conversations.json")
        return

    with open(conversations_path) as f:
        data = json.load(f)

    convs = data["conversations"]
    run_id = data.get("run_id")
    organism = (data.get("organism") or {}).get("name")

    for method in methods:
        output_path = run_dir / "analysis" / f"{method}.json"
        if output_path.exists() and not args.force:
            print(f"  Skipping {run_dir.name}/{method}: already exists (--force to overwrite)")
            continue

        print(f"  {run_dir.name} | {method}")

        if method == "cosine_sim":
            result_data = run_cosine_sim(convs, args.embedding_model)
        elif method == "vocab_entropy":
            result_data = run_vocab_entropy(convs, args.vocab_window)
        elif method == "compression":
            result_data = run_compression(convs, args.window)
        else:
            print(f"  ERROR: unknown method {method!r}", file=sys.stderr)
            continue

        output = {
            "method": method,
            "run_id": run_id,
            "organism": organism,
            "results": result_data,
            "analyzed_at": datetime.now().isoformat(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"    -> {output_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative analysis of attractor-states conversation transcripts"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="Single run directory containing conversations.json")
    target.add_argument("--experiment", type=Path, help="Experiment directory; analyze all run subdirs")
    parser.add_argument(
        "--methods", nargs="+", required=True, choices=sorted(METHODS),
        metavar="METHOD",
        help=f"Methods to run: {', '.join(sorted(METHODS))}",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for cosine_sim (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--vocab-window",
        type=int,
        default=5,
        help="Sliding window size in turns for vocab_entropy cross-turn entropy (default: 5)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Sliding window size in turns for compression (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists",
    )
    args = parser.parse_args()

    run_dirs = (
        [args.run.resolve()]
        if args.run
        else sorted(p for p in args.experiment.resolve().iterdir() if p.is_dir())
    )

    print(f"Runs: {len(run_dirs)} | Methods: {args.methods}\n")
    for run_dir in run_dirs:
        analyze_run(run_dir, args.methods, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
