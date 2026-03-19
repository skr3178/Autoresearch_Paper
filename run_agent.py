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
import subprocess
import sys
import textwrap
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
    if name == "read_file":
        return tool_read_file(args["path"])
    if name == "write_file":
        return tool_write_file(args["path"], args["content"])
    if name == "list_dir":
        return tool_list_dir(args.get("path", ""))
    if name == "read_image":
        return tool_read_image(args["path"])
    if name == "run_bash":
        return tool_run_bash(args["command"], args.get("timeout", 600))
    return f"ERROR: unknown tool {name!r}"


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
    equations = tool_read_file("paper/carplanner_equations.md")
    algorithms = tool_read_file("paper/algorithms.md")
    tables = tool_read_file("paper/tables.md")
    hyperparameters = tool_read_file("paper/hyperparameters.md")

    phase_note = ""
    if start_phase > 0:
        phase_note = textwrap.dedent(f"""
            **RESUME NOTE**: Phases 0–{start_phase - 1} are already complete. Jump directly to Phase {start_phase}.
            Do NOT re-read or re-verify earlier phases. Do NOT explore the repo structure.
            Do NOT run any git commands (no git checkout, git add, git commit, git branch).
            Before doing anything in Phase {start_phase}, check which files already exist (use list_dir) and read progress.md to see what is already done. Skip any steps whose output files already exist and are non-empty. Only work on steps that are incomplete.
            If Phase {start_phase} is Phase 3: read progress.md, find the first submodule NOT marked ✅, and start writing code for that submodule immediately.
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

        ## Python environments — TWO venvs, use the correct one every time
        - **Torch / CUDA / model training** (imports torch): `/media/skr/storage/autoresearch/.venv/bin/python <script>`
        - **nuPlan data loading** (imports nuplan.*): `/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python <script>`

        Never use `uv run python` — it has neither torch nor nuplan. Never pip install either — they are already in their respective venvs above.

        {phase_note}

        === PROGRAM (your instructions) ===
        {program}

        === REQUIREMENTS ===
        {requirements}

        === FAILURE PATTERNS (read carefully to avoid known mistakes) ===
        {failure_patterns}

        === CURRENT PAPER CONTRACT (may be empty if Phase 1 not started) ===
        {paper_contract}

        === SUBMODULES (Phase 3 build order — read before writing any model code) ===
        {submodules}

        === EQUATIONS (all numbered equations from the paper) ===
        {equations}

        === ALGORITHMS (formal pseudocode from supplementary material + reconstructed) ===
        {algorithms}

        === TABLES (all paper tables with full values) ===
        {tables}

        === HYPERPARAMETERS (all hyperparameters by module and training stage) ===
        {hyperparameters}

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
        "max_tokens": 16384,
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(start_phase: int = 0, max_turns: int = 200, dry_run: bool = False) -> None:
    # Clear any shell-inherited API vars so .env always takes precedence
    import os as _os
    for _var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AUTORESEARCH_INTELLIGENCE_MODEL"):
        _os.environ.pop(_var, None)

    load_repo_env()

    # Phase 0 needs vision — use VISION_* keys if available, else fall back to OPENAI_*
    # Currently disabled: gpt-4o is used for all phases via OPENAI_* keys in .env
    # Re-enable when switching to a coding-only model (e.g. glm-5) for phases 1+:
    # if start_phase == 0:
    #     vision_key = _os.environ.get("VISION_API_KEY")
    #     vision_url = _os.environ.get("VISION_BASE_URL")
    #     vision_model = _os.environ.get("VISION_MODEL")
    #     if vision_key:
    #         _os.environ["OPENAI_API_KEY"] = vision_key
    #     if vision_url:
    #         _os.environ["OPENAI_BASE_URL"] = vision_url
    #     if vision_model:
    #         _os.environ["AUTORESEARCH_INTELLIGENCE_MODEL"] = vision_model

    cfg = IntelligenceConfig.from_env()

    print(f"Agent starting — model: {cfg.model}  base_url: {cfg.base_url or 'https://api.z.ai/api/coding/paas/v4'}")
    print(f"Repo: {REPO_ROOT}")
    print(f"Max turns: {max_turns}\n")

    system = build_system_prompt(start_phase)

    if dry_run:
        print("=== SYSTEM PROMPT ===")
        print(system)
        return

    messages: list[dict] = [{"role": "user", "content": "Begin the implementation. Follow program.md phase by phase."}]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1}/{max_turns} ---")

        try:
            response = chat(cfg, [{"role": "system", "content": system}] + messages, TOOLS)
        except Exception as e:
            print(f"API error: {e}")
            raise

        choice = response["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")

        # Print assistant text if any
        if msg.get("content"):
            print(f"[agent] {msg['content'][:500]}")

        messages.append(msg)

        # If no tool calls, agent is done or stuck
        if not msg.get("tool_calls"):
            if finish == "stop":
                print("\n=== Agent finished ===")
            else:
                print(f"\n=== Agent stopped (finish_reason={finish!r}) ===")
            break

        # Execute tool calls
        for tc in msg["tool_calls"]:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

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
