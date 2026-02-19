"""Generate a short descriptive WandB run name from git context using the Anthropic API.

Reads git log, git diff, and a persistent history of past run titles, then asks Claude
to produce a concise title focusing on what changed since previous runs.
Appends the title and summary to a persistent append-only file.
Stores each run's git diff so future runs can compare against previous state.

Usage:
    from generate_run_name import generate_run_name
    name = generate_run_name()
"""

import json
import os
import subprocess
from datetime import UTC, datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

HISTORY_FILE = Path(__file__).parent / ".run_name_history.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(REPO_ROOT / ".env", override=True)

MAX_DIFF_CHARS = 10000
N_COMMITS = 5
HEAD_LINES_PER_FILE = 80


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _truncate(s: str, limit: int) -> str:
    if len(s) > limit:
        return s[:limit] + "\n... (truncated)"
    return s


def _current_diff() -> str:
    """Show the first `HEAD_LINES_PER_FILE` lines of each file with uncommitted changes."""
    changed_files = _git(["diff", "HEAD", "--name-only"]).splitlines()
    if not changed_files:
        return ""
    file_parts = []
    for filepath in changed_files:
        full_path = REPO_ROOT / filepath
        if not full_path.exists():
            file_parts.append(f"  {filepath}: (deleted)")
            continue
        lines = full_path.read_text().splitlines()
        head = "\n".join(lines[:HEAD_LINES_PER_FILE])
        if len(lines) > HEAD_LINES_PER_FILE:
            head += f"\n... ({len(lines) - HEAD_LINES_PER_FILE} more lines)"
        file_parts.append(f"  {filepath}:\n{head}")
    return _truncate("\n".join(file_parts), MAX_DIFF_CHARS)


def _commit_file_heads(n_commits: int, head_lines: int) -> str:
    """For each of the last n commits, show the first `head_lines` of each changed file."""
    hashes = _git(["log", "--format=%H", f"-{n_commits}"]).splitlines()
    parts = []
    for commit_hash in hashes:
        subject = _git(["log", "--format=%s", "-1", commit_hash])
        changed_files = _git(
            ["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
        ).splitlines()
        file_parts = []
        for filepath in changed_files:
            content = _git(["show", f"{commit_hash}:{filepath}"])
            lines = content.splitlines()
            head = "\n".join(lines[:head_lines])
            if len(lines) > head_lines:
                head += f"\n... ({len(lines) - head_lines} more lines)"
            file_parts.append(f"  {filepath}:\n{head}")
        parts.append(f"commit: {subject}\n" + "\n".join(file_parts))
    return "\n\n".join(parts)


def _git_context() -> str:
    branch = _git(["branch", "--show-current"])
    commit_heads = _commit_file_heads(N_COMMITS, HEAD_LINES_PER_FILE)
    diff = _current_diff()

    parts = [f"Branch: {branch}"]
    parts.append(
        f"Recent commits (first {HEAD_LINES_PER_FILE} lines of each changed file):\n{commit_heads}"
    )
    if diff:
        parts.append(f"Uncommitted changes:\n{diff}")
    return "\n\n".join(parts)


def _load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _past_context(history: list[dict]) -> tuple[str, str]:
    """Return (past_titles, past_diffs) from history."""
    if not history:
        return "(no previous runs — this is the first)", ""

    titles_parts = []
    for entry in history[-20:]:
        titles_parts.append(f"[{entry['timestamp']}] {entry['title']}")
        titles_parts.append(f"  {entry['summary']}")
    past_titles = "\n".join(titles_parts)

    recent = history[-3:]
    diff_parts = []
    for i, entry in enumerate(recent):
        n = len(history) - len(recent) + i + 1
        diff = entry.get("diff", "(no diff stored)")
        diff_parts.append(f"--- Run #{n}: {entry['title']} ---\n{diff}")
    past_diffs = "\n\n".join(diff_parts)

    return past_titles, past_diffs


def _call_api(git_context: str, past_titles: str, past_diffs: str) -> tuple[str, str]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    past_diffs_section = ""
    if past_diffs:
        past_diffs_section = f"""
Git diffs from last 3 runs (compare against current diff to see what changed):
{past_diffs}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""You title experiment runs on Weights & Biases. This is NOT a git commit message — it's a name for a training run that helps the researcher find it later in a list of runs.

The project does SPD (stochastic parameter decomposition) on multitask sparse parity models. The title should describe what experiment configuration is being run, focusing on parameter values and setup choices. Think: what would the researcher search for to find this run?

Good examples: "C=1000 pnorm=2 l* decomp", "d_mlp=100 10 tasks", "faithfulness 1000", "decompose l2 only, C=500", "imp min 0.02", "l1 only, stoch recon 0.5"
Bad examples: "coeff=0.02" (which coeff?? always say which loss/config it belongs to, e.g. "imp min 0.02")

Rules:
- Include specific parameter values from configs/code (d_mlp, C, n_control_bits, loss coefficients, module patterns, etc).
- When mentioning a coefficient or parameter, always say WHICH loss or config it belongs to. Use short names: "imp min" for ImportanceMinimalityLoss, "stoch recon" for StochasticReconSubsetLoss, "faithfulness" for FaithfulnessLoss, "PGD recon" for PGDReconSubsetLoss.
- Focus on what CHANGED vs previous runs. Don't repeat info that's constant across recent runs.
- If this is the first run or many things changed, include more context.
- If the diff is only tooling/scripts/plotting with no experiment parameter changes, title it based on whatever experiment config is visible in the code (e.g. from config files or training scripts in the patches). If you truly can't see any config, say "no config changes visible".

ENSURE YOU USE THE CORRECT d_mlp.

BE CAREFUL about code changes which may not obviously change a run parameter, but do. E.g.
```
-idx = 8
+idx = 6
```
where idx is used as an index into a list of models to train an spd run on - ENSURE YOU REFER TO THE CORRECT RUN

Past run titles and summaries (most recent last):
{past_titles}
{past_diffs_section}
Current git context:
{git_context}

Respond in exactly this format:
TITLE: <title>
SUMMARY: <one sentence on what this run's config/setup is>
""",
            }
        ],
    )
    text = response.content[0].text.strip()
    title = ""
    summary = ""
    for line in text.splitlines():
        if line.startswith("TITLE:"):
            title = line.removeprefix("TITLE:").strip()
        elif line.startswith("SUMMARY:"):
            summary = line.removeprefix("SUMMARY:").strip()
    assert title, f"Failed to parse TITLE from response:\n{text}"
    assert summary, f"Failed to parse SUMMARY from response:\n{text}"
    return title, summary


def _append_history(title: str, summary: str, diff: str) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    entry = {
        "timestamp": timestamp,
        "title": title,
        "summary": summary,
        "diff": diff,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_run_name() -> str:
    git_context = _git_context()
    diff = _current_diff()
    history = _load_history()
    past_titles, past_diffs = _past_context(history)
    title, summary = _call_api(git_context, past_titles, past_diffs)
    _append_history(title, summary, diff)
    return title


if __name__ == "__main__":
    print(generate_run_name())
