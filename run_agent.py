"""
Autonomous research paper implementation agent.

Uses the z.ai (OpenAI-compatible) API to drive the full implementation
pipeline defined in program.md — from paper extraction through training.

Usage:
    python run_agent.py
    python run_agent.py --phase 1          # start from a specific phase
    python run_agent.py --max-turns 50     # limit turns (default: 200)
    python run_agent.py --dry-run          # print system prompt and exit

Environment (.env):
    OPENAI_API_KEY        — API key
    OPENAI_BASE_URL       — base URL (e.g. https://api.z.ai/api/coding/paas/v4)
    AUTORESEARCH_INTELLIGENCE_MODEL — model name (e.g. glm-5, gpt-4o)

Phase 0 requires a vision-capable model (e.g. gpt-4o) to annotate figures.
Phases 1–6 work with any strong coding model (e.g. glm-5).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from intelligence_config import IntelligenceConfig, load_repo_env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 32_000   # truncate large files to keep context manageable
MAX_BASH_BYTES = 16_000   # truncate large command output

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Sentinel prefix used to identify image tool results in the message list
_IMAGE_SENTINEL = "__IMAGE__:"

# API retry settings
API_RETRY_COUNT = 3
API_RETRY_WAIT = 15  # seconds between retries on timeout

# Max total chars across all messages before pruning old tool results
MAX_HISTORY_CHARS = 20_000

# Python code patterns that should not appear in reasoning text
_CODE_CORRUPTION_RE = re.compile(
    r'^\s*(def |class |import |from \w+ import|if |for |while |return )'
    r'|torch\.\w+\(|nn\.\w+\(|\.cuda\(\)|\.to\(device',
    re.MULTILINE,
)


def _is_corrupted(content: str | None) -> bool:
    """Return True if agent reasoning text contains Python code fragments."""
    if not content:
        return False
    return bool(_CODE_CORRUPTION_RE.search(content))


def _get_newly_completed_submodules(old_text: str, new_text: str) -> list[str]:
    """Return submodule display names that gained ✅ between old and new progress.md."""
    def _extract(text: str) -> set[str]:
        return set(re.findall(r'\*\*([^*]+)\*\*\s*✅', text))
    return list(_extract(new_text) - _extract(old_text))


def _has_figure_verification(proof_text: str, submodule_name: str) -> bool:
    """Return True if proof.md contains a real figure verification for this submodule.

    Minimum requirements:
    1. A "Figure Verification" header exists in the submodule's section.
    2. At least one figure classified as "architecture" or "results".

    If any architecture figure is referenced for this submodule, ALSO require:
    3. At least one forward mapping:  Figure N, component "..." → file.py:ClassName
    4. At least one reverse mapping:  file.py:ClassName → Figure N, component "..."

    If all relevant figures are classified as "results" (no architecture figures),
    the submodule passes with just the header + classification — there is nothing
    to compare against code.
    """
    n = submodule_name.lower()
    name_variants = [
        n,
        n.replace(" ", "_"),
        n.replace(" ", ""),
        n.replace("_", " "),
        n.replace("-", "").replace(" ", "_"),   # "Auto-regressive Policy" → "autoregressive_policy"
        n.replace("-", "").replace(" ", ""),    # "Auto-regressive Policy" → "autoregressivepolicy"
        n.replace("-", " "),                    # "auto-regressive" → "auto regressive"
    ]
    # Split into sections by "## "
    sections = re.split(r'(?m)^## ', proof_text)
    for section in sections:
        first_line = section.split("\n", 1)[0].lower()
        if any(v in first_line for v in name_variants):
            # 1. Must have an explicit "Figure Verification" subsection
            has_header = bool(re.search(r'figure\s+verification', section, re.IGNORECASE))
            if not has_header:
                return False

            # 2. At least one figure classified as "architecture" or "results"
            # Matches lines like: "figure2_page3.png: architecture (full diagram)"
            has_classification = bool(re.search(
                r':\s*(architecture|results)',
                section,
                re.IGNORECASE,
            ))
            if not has_classification:
                return False

            # Check if any architecture figure is referenced
            has_architecture = bool(re.search(
                r'architecture\s*(figure|→|:|—|-)',
                section,
                re.IGNORECASE,
            ))

            # If architecture figures exist, require bidirectional mapping
            if has_architecture:
                has_forward = bool(re.search(
                    r'[Ff]igure\s*\d+.*→.*\.py',
                    section,
                ))
                has_reverse = bool(re.search(
                    r'\.py\s*:\s*\w+.*→.*[Ff]igure\s*\d+',
                    section,
                ))
                return has_forward and has_reverse

            # No architecture figures — results-only submodule passes with
            # just the header + classification
            return True
    return False


def _prune_history(messages: list[dict]) -> list[dict]:
    """Drop oldest assistant+tool blocks when total history exceeds MAX_HISTORY_CHARS.

    Always removes complete blocks (assistant message with tool_calls + all its
    tool responses) to avoid orphaned tool_call_ids that cause API 400 errors.
    """
    def _size(m: dict) -> int:
        c = m.get("content") or ""
        return len(c) if isinstance(c, str) else sum(len(x.get("text", "")) for x in c if isinstance(x, dict))

    if _size_total(messages) <= MAX_HISTORY_CHARS:
        return messages

    pruned = list(messages)
    while _size_total(pruned) > MAX_HISTORY_CHARS and len(pruned) > 10:
        # Find the oldest assistant message that has tool_calls
        removed = False
        for i, m in enumerate(pruned):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                # Collect all tool_call_ids from this message
                ids = {tc["id"] for tc in m["tool_calls"]}
                # Remove the assistant message
                pruned.pop(i)
                # Remove all corresponding tool response messages
                pruned = [
                    msg for msg in pruned
                    if not (msg.get("role") == "tool" and msg.get("tool_call_id") in ids)
                ]
                removed = True
                break
        if not removed:
            break  # nothing left to prune safely
    return pruned


def _size_total(messages: list[dict]) -> int:
    def _size(m: dict) -> int:
        c = m.get("content") or ""
        return len(c) if isinstance(c, str) else sum(len(x.get("text", "")) for x in c if isinstance(x, dict))
    return sum(_size(m) for m in messages)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository. Returns file contents as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root or absolute."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root or absolute."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to list. Defaults to repo root."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": (
                "Read an image file and return its contents for visual inspection. "
                "Use this to examine figures, diagrams, and plots extracted from the paper. "
                "Supports PNG, JPG, JPEG, GIF, WEBP."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file, relative to repo root or absolute."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a bash command in the repo root. "
                "Use for: git, uv run, python, grep, etc. "
                "Working directory is always the repo root. "
                "Timeout: 600s. Stdout and stderr are returned together."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 600, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def tool_read_file(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR reading {p}: {e}"
    if len(content) > MAX_FILE_BYTES:
        content = content[:MAX_FILE_BYTES] + f"\n... [truncated at {MAX_FILE_BYTES} bytes]"
    return content


def tool_write_file(path: str, content: str) -> str:
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"ERROR writing {p}: {e}"


def tool_list_dir(path: str = "") -> str:
    p = _resolve(path) if path else REPO_ROOT
    if not p.exists():
        return f"ERROR: path not found: {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            tag = "DIR " if e.is_dir() else "FILE"
            lines.append(f"{tag}  {e.name}")
        return "\n".join(lines) if lines else "(empty)"
    except Exception as e:
        return f"ERROR listing {p}: {e}"


def tool_read_image(path: str) -> str:
    """Returns a sentinel string encoding the image as base64 data URL."""
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    ext = p.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        return f"ERROR: unsupported image type {ext!r}. Supported: {sorted(IMAGE_EXTENSIONS)}"
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime = MIME_TYPES[ext]
        # Sentinel format: __IMAGE__:<mime>;<base64data>
        return f"{_IMAGE_SENTINEL}{mime};{b64}"
    except Exception as e:
        return f"ERROR reading image {p}: {e}"


def tool_run_bash(command: str, timeout: int = 600) -> str:
    timeout = min(int(timeout), 600)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout + result.stderr
        if len(out) > MAX_BASH_BYTES:
            out = out[:MAX_BASH_BYTES] + f"\n... [truncated at {MAX_BASH_BYTES} bytes]"
        rc = result.returncode
        return f"exit_code={rc}\n{out}"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            return tool_read_file(args["path"])
        if name == "write_file":
            if "content" not in args:
                return "ERROR: write_file called without 'content' argument — your tool call was malformed (likely due to a truncated API response). Re-issue the write_file call with the full file content."
            return tool_write_file(args["path"], args["content"])
        if name == "list_dir":
            return tool_list_dir(args.get("path", ""))
        if name == "read_image":
            return tool_read_image(args["path"])
        if name == "run_bash":
            if "command" not in args:
                return "ERROR: run_bash called without 'command' argument — your tool call was malformed. Re-issue with the full command string."
            return tool_run_bash(args["command"], args.get("timeout", 600))
        return f"ERROR: unknown tool {name!r}"
    except KeyError as e:
        return f"ERROR: tool {name!r} missing required argument {e} — your tool call was malformed (likely truncated API response). Re-issue the tool call with all required arguments."


def _make_tool_result_content(result: str) -> list | str:
    """Convert a tool result string into the appropriate content format.

    If the result is an image sentinel, return a multimodal content list
    (text + image_url) for vision-capable models. Otherwise return plain string.
    """
    if result.startswith(_IMAGE_SENTINEL):
        payload = result[len(_IMAGE_SENTINEL):]  # "mime;base64data"
        mime, _, b64 = payload.partition(";")
        return [
            {"type": "text", "text": "Image loaded:"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    return result


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(start_phase: int) -> str:
    program = tool_read_file("program.md")
    requirements = tool_read_file("requirements.md")
    failure_patterns = tool_read_file("failure_patterns.md")
    paper_contract = tool_read_file("paper_contract.md")
    submodules = tool_read_file("submodules.md")
    # equations, algorithms, tables, hyperparameters are NOT pre-loaded —
    # the agent must actively read them via read_file/read_image as instructed in program.md Step 1.

    phase_note = ""
    if start_phase > 0:
        phase_note = textwrap.dedent(f"""
            **RESUME NOTE**: Phases 0–{start_phase - 1} are already complete. Jump directly to Phase {start_phase}.
            Do NOT re-read or re-verify earlier phases. Do NOT explore the repo structure.
            Do NOT run any git commands (no git checkout, git add, git commit, git branch).
            Before doing anything in Phase {start_phase}, check which files already exist (use list_dir) and read progress.md to see what is already done. Skip any steps whose output files already exist and are non-empty. Only work on steps that are incomplete.
            If Phase {start_phase} is Phase 3: read progress.md, find the first submodule NOT marked ✅, and start writing code for that submodule immediately.
            Do NOT stop after completing one submodule. After each submodule gate passes and progress.md is updated, immediately move to the next submodule. Only stop when ALL 9 submodules are marked ✅ in progress.md.
        """).strip()

    return textwrap.dedent(f"""
        You are an autonomous research paper implementation agent.
        Your job is to implement the research paper described in requirements.md,
        following the step-by-step phases in program.md exactly.

        You have five tools:
        - read_file: read any text file in the repo
        - read_image: read an image file for visual inspection — use this to inspect paper figures directly
        - write_file: create or update any file
        - list_dir: list directory contents
        - run_bash: run shell commands (git, python, uv run, grep, etc.)

        ## Reading paper figures
        Paper figures are in paper/images/ as PNG files. You are a multimodal model — always use
        read_image to inspect figures directly. Do NOT use read_file on .txt annotation files as
        a substitute; read the raw image for full fidelity. Use read_image for every figure before
        writing any architectural decisions that depend on it.

        Working directory for run_bash is always: {REPO_ROOT}

        ## Python environment — ONE venv for everything
        Use this python for ALL scripts (data loading, training, testing):
        `/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python <script>`

        This venv has: torch 1.9.0+cu111, nuplan-devkit, numpy, CUDA support.
        Never use `uv run python` or `/media/skr/storage/autoresearch/.venv/bin/python`.
        Never pip install anything — all packages are already installed.

        {phase_note}

        === PROGRAM (your instructions) ===
        {program}

        === REQUIREMENTS ===
        {requirements}

        === FAILURE PATTERNS (read carefully to avoid known mistakes) ===
        {failure_patterns}

        === CURRENT PAPER CONTRACT ===
        {paper_contract}

        === SUBMODULES (Phase 3 build order — read before writing any model code) ===
        {submodules}

        ## Paper artifacts — read actively via tools, do NOT rely on memory
        The following files are NOT pre-loaded. You MUST read them explicitly before each submodule:
        - paper/carplanner_equations.md — all paper equations
        - paper/algorithms.md — all pseudocode
        - paper/hyperparameters.md — all hyperparameter values
        - paper/tables.md — all tables
        - paper/images/*.png — all figures (use read_image)
        - paper/images/*.txt — figure annotations

        ---
        Begin now. Follow every phase and exit gate exactly as written in the program.
        Do not skip phases. Commit after each phase as instructed.
        When you are completely done (all phases passed), output the final summary and stop.
    """).strip()


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------

def chat(cfg: IntelligenceConfig, messages: list[dict], tools: list[dict]) -> dict:
    import urllib.request

    body = {
        "model": cfg.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_completion_tokens": 4096,
        "temperature": 0.3,
    }
    payload = json.dumps(body).encode()
    base_url = (cfg.base_url or "https://api.z.ai/api/coding/paas/v4").rstrip("/")
    url = f"{base_url}/chat/completions"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _get_paper_image_paths() -> set[str]:
    """Return the set of normalised absolute paths for all paper images."""
    img_dir = REPO_ROOT / "paper" / "images"
    if not img_dir.exists():
        return set()
    return {str(p.resolve()) for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


def _write_paper_context_sentinel(reads: set[str]) -> None:
    """Write paper_context.md so the agent knows all figures have been loaded."""
    names = sorted(Path(p).name for p in reads)
    content = (
        "# Paper Context\n\n"
        "All paper figures have been read this session.\n"
        "Do NOT call read_image again — the images are already loaded in your context.\n\n"
        "Figures read:\n" + "\n".join(f"- {n}" for n in names) + "\n"
    )
    (REPO_ROOT / "paper_context.md").write_text(content, encoding="utf-8")
    print(f"[runner] Auto-wrote paper_context.md after reading all {len(reads)} paper figures.")


def run_agent(start_phase: int = 0, max_turns: int = 200, dry_run: bool = False) -> None:
    # Clear any shell-inherited API vars so .env always takes precedence
    import os as _os
    for _var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AUTORESEARCH_INTELLIGENCE_MODEL"):
        _os.environ.pop(_var, None)

    load_repo_env()

    cfg = IntelligenceConfig.from_env()

    print(f"Agent starting — model: {cfg.model}  base_url: {cfg.base_url or 'https://api.z.ai/api/coding/paas/v4'}")
    print(f"Repo: {REPO_ROOT}")
    print(f"Max turns: {max_turns}\n")

    system = build_system_prompt(start_phase)

    if dry_run:
        print("=== SYSTEM PROMPT ===")
        print(system)
        return

    # Remove session sentinel so agent re-reads paper artifacts on each fresh run
    (REPO_ROOT / "paper_context.md").unlink(missing_ok=True)

    # Session-level dedup cache for read_image.
    # Key: normalised absolute path → base result string (already returned once).
    # On re-reads we return a short stub to avoid flooding history with base64 blobs.
    session_image_reads: dict[str, str] = {}
    all_paper_images = _get_paper_image_paths()

    # Session-level dedup cache for paper annotation .txt files.
    # These are small but re-read every cycle after a gate block, bloating history.
    # Key: normalised absolute path → True (already returned full content once).
    session_txt_reads: set[str] = set()
    _PAPER_IMAGES_DIR = str((REPO_ROOT / "paper" / "images").resolve())

    # Snapshot of progress.md used by the hard gate to detect newly added ✅ marks
    prev_progress = tool_read_file("progress.md")

    messages: list[dict] = [{"role": "user", "content": "Begin the implementation. Follow program.md phase by phase."}]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1}/{max_turns} ---")

        # Prune history to avoid context bloat causing API timeouts
        messages = _prune_history(messages)

        # API call with retry on timeout
        response = None
        for attempt in range(API_RETRY_COUNT):
            try:
                response = chat(cfg, [{"role": "system", "content": system}] + messages, TOOLS)
                break
            except TimeoutError as e:
                if attempt < API_RETRY_COUNT - 1:
                    print(f"API timeout (attempt {attempt + 1}/{API_RETRY_COUNT}), retrying in {API_RETRY_WAIT}s...")
                    time.sleep(API_RETRY_WAIT)
                else:
                    print(f"API error: {e}")
                    raise
            except Exception as e:
                print(f"API error: {e}")
                raise

        choice = response["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")

        # Detect output corruption (Python code in reasoning text)
        content = msg.get("content") or ""
        if _is_corrupted(content):
            print(f"[runner] Output corruption detected — injecting recovery prompt")
            msg_clean = dict(msg)
            msg_clean["content"] = "[response contained code fragments and was truncated by runner]"
            messages.append(msg_clean)
            # If the corrupted message had tool_calls, add dummy responses so the
            # message history stays valid (OpenAI requires every tool_call_id to
            # have a corresponding tool response before the next user message)
            for tc in msg_clean.get("tool_calls") or []:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "[tool call skipped due to output corruption]",
                })
            messages.append({
                "role": "user",
                "content": (
                    "Your last response contained Python code fragments in the reasoning text outside any tool call. "
                    "This causes context corruption. Recover now: in plain text only, state in ONE sentence what the "
                    "current failing test is and what you will do next. Then make exactly one tool call."
                ),
            })
            continue

        # Print assistant text if any
        if content:
            print(f"[agent] {content[:500]}")

        messages.append(msg)

        # If no tool calls, check if phase is truly complete before stopping
        if not msg.get("tool_calls"):
            if finish == "stop" and start_phase == 3:
                # Check progress.md — only stop if all Phase 3 submodules are ✅
                progress = tool_read_file("progress.md")
                if "Phase 3" in progress and progress.count("✅") >= progress.count("⬜") + progress.count("⏳") + progress.count("✅"):
                    # Rough check: see if any submodule is NOT marked done
                    import re as _re
                    incomplete = _re.findall(r'⬜|⏳|IN PROGRESS|TODO', progress)
                    if incomplete:
                        print(f"[runner] Agent stopped early — {len(incomplete)} incomplete item(s) in progress.md. Injecting continuation prompt.")
                        messages.append({
                            "role": "user",
                            "content": (
                                "You stopped but Phase 3 is not complete. "
                                "Read progress.md, find the next submodule NOT marked ✅, and implement it now. "
                                "Do not stop until ALL submodules in submodules.md are marked ✅."
                            ),
                        })
                        continue
                print("\n=== Agent finished ===")
            elif finish == "stop":
                print("\n=== Agent finished ===")
            else:
                print(f"\n=== Agent stopped (finish_reason={finish!r}) ===")
            break

        # Execute tool calls
        context_ready_injected = False
        for tc in msg["tool_calls"]:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            # --- read_file dedup for paper annotation .txt files ---
            # After a gate block, the agent re-reads all 6 annotation files every cycle,
            # bloating history and causing pruning that wipes the gate message.
            if fn == "read_file":
                path_arg = args.get("path", "")
                norm_txt = str(_resolve(path_arg))
                is_annotation = (
                    norm_txt.endswith(".txt")
                    and norm_txt.startswith(_PAPER_IMAGES_DIR)
                )
                if is_annotation and norm_txt in session_txt_reads:
                    result = (
                        f"[Runner: annotation already read this session — stub returned. "
                        f"Do NOT re-read paper annotation files. "
                        f"Focus on writing the Figure Verification section to proof.md.]"
                    )
                    print(f"  TOOL {fn}({path_arg!r}) → [annotation cached — skipping re-read]")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _make_tool_result_content(result),
                    })
                    continue
                elif is_annotation:
                    print(f"  TOOL {fn}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
                    result = execute_tool(fn, args)
                    session_txt_reads.add(norm_txt)
                    preview = result[:200].replace("\n", " ")
                    print(f"       → {preview}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _make_tool_result_content(result),
                    })
                    continue

            # --- read_image dedup ---
            # Images are large base64 blobs. Re-reading them fills the history,
            # triggers pruning, which erases the read evidence, causing infinite loops.
            # Solution: cache first read; return a short stub on subsequent calls.
            if fn == "read_image":
                norm = str(_resolve(args.get("path", "")))
                if norm in session_image_reads:
                    result = (
                        "[Runner: image already loaded this session — returning cached stub to preserve context. "
                        "Do NOT call read_image for this file again. "
                        "paper_context.md lists all figures read so far.]"
                    )
                    print(f"  TOOL {fn}({args.get('path', '')!r}) → [cached — skipping re-read]")
                else:
                    print(f"  TOOL {fn}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
                    result = execute_tool(fn, args)
                    session_image_reads[norm] = "read"
                    print(f"       → [image: {args.get('path', '?')}]")
                    # After reading all paper images for the first time, auto-write sentinel
                    if (not (REPO_ROOT / "paper_context.md").exists()
                            and all_paper_images
                            and all_paper_images.issubset(session_image_reads.keys())):
                        _write_paper_context_sentinel(set(session_image_reads.keys()))
                        context_ready_injected = True
            else:
                print(f"  TOOL {fn}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
                result = execute_tool(fn, args)
                # Images get a short preview; text results get first 200 chars
                if result.startswith(_IMAGE_SENTINEL):
                    print(f"       → [image: {args.get('path', '?')}]")
                else:
                    preview = result[:200].replace("\n", " ")
                    print(f"       → {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": _make_tool_result_content(result),
            })

        # --- Hard gate: figure verification must exist before ✅ ---
        # Check if any write_file call this turn targeted progress.md
        progress_written_this_turn = any(
            tc["function"]["name"] == "write_file"
            and "progress" in json.loads(tc["function"].get("arguments", "{}")).get("path", "")
            for tc in msg["tool_calls"]
        )
        if progress_written_this_turn:
            new_progress = tool_read_file("progress.md")
            newly_done = _get_newly_completed_submodules(prev_progress, new_progress)
            gate_failures = []
            images_not_read = not session_image_reads  # no images read at all this session
            if newly_done:
                if images_not_read:
                    # Agent hasn't read any figures this session — can't have done verification
                    gate_failures = list(newly_done)
                else:
                    proof_text = tool_read_file("proof.md")
                    for name in newly_done:
                        if not _has_figure_verification(proof_text, name):
                            gate_failures.append(name)
            if gate_failures:
                # Revert progress.md — the agent has not earned these ✅ marks
                tool_write_file("progress.md", prev_progress)
                failed = ", ".join(f'"{n}"' for n in gate_failures)
                reason = (
                    "you have NOT called read_image on any paper figures this session"
                    if images_not_read else
                    "proof.md is missing the required Figure Verification section with bidirectional mapping"
                )
                print(f"[runner] HARD GATE blocked ✅ for: {failed} — {reason}. Reverting progress.md.")
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Runner] HARD GATE BLOCKED: You marked {failed} as ✅ in progress.md "
                        f"but {reason}. progress.md has been reverted.\n\n"
                        f"⚠️ DO NOT modify any .py files. DO NOT re-read paper figures or annotations — "
                        f"you have already read them. DO NOT read progress.md, paper_context.md, or "
                        f"annotation .txt files again.\n\n"
                        f"ACTION REQUIRED — do this NOW in one step:\n"
                        f"Open proof.md, find the section for each blocked submodule, "
                        f"and append a '### Figure Verification' block in this EXACT format:\n\n"
                        f"### Figure Verification\n"
                        f"**Figure classification**\n"
                        f"- figureN_pageX.png: architecture (brief reason)\n"
                        f"- figureM_pageY.png: results (brief reason)\n\n"
                        f"**Forward mapping (figure → code)**\n"
                        f"- Figure N, component \"ComponentName\" → implementation/file.py:ClassName\n\n"
                        f"**Reverse mapping (code → figure)**\n"
                        f"- implementation/file.py:ClassName → Figure N, component \"ComponentName\"\n\n"
                        f"Write this section based on what you already know from the figures and code. "
                        f"Then re-run tests and update progress.md."
                    ),
                })
            else:
                prev_progress = new_progress  # gate passed — advance snapshot

        # After all paper images have been read for the first time, inject a prompt
        # so the agent knows to stop reading and start writing code.
        if context_ready_injected:
            messages.append({
                "role": "user",
                "content": (
                    "[Runner] All paper figures have been loaded into context. "
                    "paper_context.md has been written — you do NOT need to call read_image again. "
                    "Stop reading paper artifacts. Write paper_context.md to confirm, then immediately "
                    "start writing implementation code for the next incomplete submodule."
                ),
            })

    else:
        print(f"\n=== Reached max turns ({max_turns}) ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous paper implementation agent")
    parser.add_argument("--phase", type=int, default=0, help="Start from this phase (0=beginning)")
    parser.add_argument("--max-turns", type=int, default=200, help="Max agent turns (default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Print system prompt and exit")
    args = parser.parse_args()

    run_agent(start_phase=args.phase, max_turns=args.max_turns, dry_run=args.dry_run)
