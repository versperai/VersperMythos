"""
VersperMythos TUI — Entry point.

Two modes:
  uv run verspermythos tui              → Launcher mode: spawns React TUI, then exits
  uv run python -m src.tui --backend-only  → Backend mode: serves as JSON-lines backend
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_bun() -> str:
    bun = shutil.which("bun")
    if bun:
        return bun
    print("Error: bun not found. Install it from https://bun.sh", file=sys.stderr)
    sys.exit(1)


def resolve_tui_dir() -> str:
    """Find the tui/ frontend directory."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "tui",
        Path(__file__).resolve().parent.parent / "tui",
    ]
    for c in candidates:
        if (c / "package.json").exists():
            return str(c)
    print("Error: tui/ frontend directory not found", file=sys.stderr)
    sys.exit(1)


def launch_tui() -> None:
    """Launcher mode: spawn React TUI with backend command in env."""
    tui_dir = resolve_tui_dir()
    bun = find_bun()

    # Build the backend command that the React TUI will spawn.
    # Use -m tui.__main__ since the editable install puts src/ on sys.path.
    backend_cmd = [sys.executable, "-m", "tui.__main__", "--backend-only"]

    # The project root so the backend finds model weights relative to it,
    # not relative to bun's CWD (which is set to tui/ by subprocess.run).
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    config = json.dumps({
        "backend_command": backend_cmd,
        "cwd": project_root,
    })
    env = os.environ.copy()
    env["VM_TUI_CONFIG"] = config

    # Run bun to start the React TUI
    proc = subprocess.run(
        [bun, "run", "src/index.tsx"],
        cwd=tui_dir,
        env=env,
    )
    sys.exit(proc.returncode)


def run_backend() -> None:
    """Backend-only mode: serve JSON-lines protocol."""
    _root = Path(__file__).resolve().parent.parent.parent

    # Ensure project root is on sys.path so from src.lm / src.vlm / src.rlm
    # imports work (the editable install .pth only adds src/ itself).
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # Always run from the project root so relative paths like "out/" and
    # "model/tokenizer" resolve correctly regardless of spawner CWD.
    os.chdir(str(_root))

    import asyncio
    from .backend_host import BackendHost
    host = BackendHost()
    asyncio.run(host.run())


def main() -> None:
    parser = argparse.ArgumentParser(prog="verspermythos")
    parser.add_argument("--backend-only", action="store_true", help="Run as backend host (internal)")
    sub = parser.add_subparsers(dest="command")

    tui_parser = sub.add_parser("tui", help="Launch the terminal UI")
    tui_parser.add_argument("--data-dir", default="./datasets")
    tui_parser.add_argument("--out-dir", default="./out")

    args = parser.parse_args()

    if args.backend_only:
        run_backend()
    elif args.command == "tui":
        launch_tui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
