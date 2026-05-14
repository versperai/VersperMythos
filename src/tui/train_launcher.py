"""Training subprocess launcher for TUI."""

import asyncio
from typing import Any


async def launch_training(cmd: list[str]) -> dict[str, Any]:
    """Launch training as a subprocess.

    The cmd should be something like:
        ["python", "-m", "src.tui.bin.train_lm", "--config", "..."]
    """
    if not cmd:
        return {"error": "No command provided"}
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    return {"status": "launched", "pid": process.pid}
