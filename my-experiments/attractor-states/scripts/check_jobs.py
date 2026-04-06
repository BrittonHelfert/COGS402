#!/usr/bin/env python3
"""
Check status and timing of submitted attractor-states SLURM jobs.

Usage:
  uv run python scripts/check_jobs.py                        # latest manifest
  uv run python scripts/check_jobs.py --manifest PATH        # specific manifest
  uv run python scripts/check_jobs.py --update-configs       # update model YAML time_limits
  uv run python scripts/check_jobs.py --update-configs --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── Time helpers ───────────────────────────────────────────────────────────────

def parse_time(s: str) -> int:
    """Parse [D-]HH:MM:SS to seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    h, m, sec = (int(x) for x in s.split(":"))
    return days * 86400 + h * 3600 + m * 60 + sec


def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def round_up_30min(seconds: int) -> int:
    interval = 30 * 60
    return ((seconds + interval - 1) // interval) * interval


# ── Manifest ───────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> list[tuple[str, str]]:
    """Returns [(job_id, label), ...]."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))
    return entries


def find_latest_manifest() -> Path:
    manifest = ROOT / "logs" / "submission_manifest.txt"
    if manifest.exists():
        return manifest
    raise FileNotFoundError(f"No manifest found at {manifest}")


# ── sacct ──────────────────────────────────────────────────────────────────────

def query_sacct(job_ids: list[str]) -> dict[str, dict]:
    """Returns {job_id: {state, elapsed_s, timelimit_s}}."""
    if not job_ids:
        return {}

    result = subprocess.run(
        [
            "sacct", "-j", ",".join(job_ids),
            "--format=JobID,State,Elapsed,Timelimit",
            "--noheader", "--parsable2",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: sacct failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    records: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, state, elapsed, timelimit = parts[0], parts[1], parts[2], parts[3]
        if "." in job_id:  # skip job steps (123456.batch, etc.)
            continue
        state_clean = state.split()[0] if state else state
        try:
            elapsed_s = parse_time(elapsed) if elapsed and elapsed not in ("None", "") else None
            timelimit_s = (
                parse_time(timelimit)
                if timelimit and timelimit not in ("None", "Partition_Limit", "")
                else None
            )
        except (ValueError, IndexError):
            elapsed_s = timelimit_s = None
        records[job_id] = {"state": state_clean, "elapsed_s": elapsed_s, "timelimit_s": timelimit_s}

    return records


# ── Model config helpers ───────────────────────────────────────────────────────

def load_model_keys() -> set[str]:
    keys = set()
    for yaml_file in (ROOT / "configs" / "models").glob("*.yaml"):
        for line in yaml_file.read_text().splitlines():
            if line.startswith("name:"):
                keys.add(line.split(":", 1)[1].strip())
    return keys


def get_model_key(label: str, model_keys: set[str]) -> str | None:
    """Extract model key from a label like 'em_bad_medical_advice_llama31_8b'."""
    for key in model_keys:
        if label == key or label.endswith("_" + key):
            return key
    return None


def read_current_time_limit(model_key: str) -> str | None:
    yaml_file = ROOT / "configs" / "models" / f"{model_key}.yaml"
    for line in yaml_file.read_text().splitlines():
        if line.startswith("time_limit:"):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def update_model_yaml(model_key: str, new_time: str, dry_run: bool) -> None:
    yaml_file = ROOT / "configs" / "models" / f"{model_key}.yaml"
    content = yaml_file.read_text()
    if re.search(r"^time_limit:", content, re.MULTILINE):
        new_content = re.sub(
            r'^time_limit:.*$', f'time_limit: "{new_time}"', content, flags=re.MULTILINE
        )
    else:
        new_content = re.sub(
            r'^(gpu_count:.*\n)', rf'\1time_limit: "{new_time}"\n', content, flags=re.MULTILINE
        )
    if dry_run:
        print(f"  [dry-run] {yaml_file.name}: time_limit → \"{new_time}\"")
    else:
        yaml_file.write_text(new_content)
        print(f"  Updated {yaml_file.name}: time_limit → \"{new_time}\"")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Check SLURM job timing for attractor-states runs")
    parser.add_argument("--manifest", type=Path, help="Path to submission manifest (default: logs/submission_manifest.txt)")
    parser.add_argument("--update-configs", action="store_true", help="Update model YAML time_limits from observed runtimes")
    parser.add_argument("--dry-run", action="store_true", help="With --update-configs: show proposed changes without writing")
    parser.add_argument("--padding", type=float, default=0.25, help="Overhead fraction to add when calibrating (default: 0.25)")
    args = parser.parse_args()

    manifest_path = args.manifest or find_latest_manifest()
    print(f"Manifest: {manifest_path}\n")

    entries = load_manifest(manifest_path)
    if not entries:
        print("No jobs in manifest.")
        return

    job_ids = [e[0] for e in entries]
    sacct_data = query_sacct(job_ids)
    model_keys = load_model_keys()

    print(f"{'JOB':>10}  {'STATE':<12}  {'ELAPSED':>8}  {'LIMIT':>8}  {'UTIL%':>6}  LABEL")
    print("─" * 85)

    timeouts: list[str] = []
    model_elapsed: dict[str, list[int]] = {}

    for job_id, label in entries:
        rec = sacct_data.get(job_id)
        if rec is None:
            state, elapsed_str, limit_str, util_str = "UNKNOWN", "-", "-", "-"
        else:
            state = rec["state"]
            elapsed_str = format_time(rec["elapsed_s"]) if rec["elapsed_s"] is not None else "-"
            limit_str = format_time(rec["timelimit_s"]) if rec["timelimit_s"] is not None else "-"
            if rec["elapsed_s"] is not None and rec["timelimit_s"] and rec["timelimit_s"] > 0:
                util_str = f"{100 * rec['elapsed_s'] / rec['timelimit_s']:.0f}%"
            else:
                util_str = "-"

        flag = "  *** TIMEOUT ***" if state == "TIMEOUT" else ""
        print(f"{job_id:>10}  {state:<12}  {elapsed_str:>8}  {limit_str:>8}  {util_str:>6}  {label}{flag}")

        if state == "TIMEOUT":
            timeouts.append(label)

        if state == "COMPLETED" and rec and rec["elapsed_s"] is not None:
            key = get_model_key(label, model_keys)
            if key:
                model_elapsed.setdefault(key, []).append(rec["elapsed_s"])

    print()

    if timeouts:
        print(f"TIMEOUTS ({len(timeouts)}):")
        for t in timeouts:
            print(f"  {t}")
        print()

    if args.update_configs:
        if not model_elapsed:
            print("No COMPLETED runs found — nothing to calibrate.")
            return
        print(f"Calibrated limits (max observed + {args.padding:.0%} padding, rounded to 30 min):")
        for key, times in sorted(model_elapsed.items()):
            max_s = max(times)
            new_s = round_up_30min(int(max_s * (1 + args.padding)))
            new_time = format_time(new_s)
            current = read_current_time_limit(key)
            if current is None:
                direction = "new"
            else:
                current_s = parse_time(current)
                direction = "↑" if new_s > current_s else ("↓" if new_s < current_s else "=")
            print(f"  {key}: max={format_time(max_s)}  current={current or '?'}  {direction}  proposed={new_time}")
            update_model_yaml(key, new_time, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
