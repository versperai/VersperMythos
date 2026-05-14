"""
VersperMythos TUI — Protocol models.

Frontend↔Backend communication via JSON-lines over stdin/stdout.
Backend→Frontend: VMJSON:<json>\n
Frontend→Backend: <json>\n
"""

from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel


class FrontendRequest(BaseModel):
    type: Literal[
        "discover",
        "load_model",
        "generate",
        "stop_generation",
        "eda_scan",
        "eda_stats",
        "eda_sample",
        "train_launch",
        "train_stop",
        "train_status",
        "bench_run",
        "cancel",
        "shutdown",
    ]
    model_type: str | None = None          # lm | vlm | rlm
    weight: str | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = 8
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    use_moe: bool | None = False
    dim: int | None = None
    prompt: str | None = None
    image_path: str | None = None
    max_new_tokens: int | None = 512
    temperature: float | None = 0.7
    top_p: float | None = 0.85
    data_dir: str | None = None
    file_path: str | None = None
    index: int | None = 0
    bench_name: str | None = None
    train_cmd: list[str] | None = None


class BackendEvent(BaseModel):
    type: Literal[
        "ready",
        "model_list",
        "model_loaded",
        "token",
        "done",
        "error",
        "metrics",
        "eda_result",
        "eda_stats_result",
        "eda_sample_result",
        "bench_result",
        "train_log",
        "train_status",
        "shutdown",
    ]
    message: str | None = None
    data: Any | None = None
