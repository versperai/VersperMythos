"""
VersperMythos TUI — Backend Host.

Reads JSON requests from stdin, processes them,
emits VMJSON: prefixed events to stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import signal
from typing import Any

from .protocol import FrontendRequest, BackendEvent

_PROTOCOL_PREFIX = "VMJSON:"


class BackendHost:
    def __init__(self):
        self._request_queue: asyncio.Queue[FrontendRequest] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._running = True
        self._model = None
        self._tokenizer = None
        self._train_process: asyncio.subprocess.Process | None = None

    # ── emit / send ────────────────────────────────────────────

    async def emit(self, event: BackendEvent) -> None:
        async with self._write_lock:
            payload = _PROTOCOL_PREFIX + event.model_dump_json(exclude_none=True) + "\n"
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write(payload.encode("utf-8"))
                buf.flush()
            else:
                sys.stdout.write(payload)
                sys.stdout.flush()

    # ── request loop ──────────────────────────────────────────

    async def _read_requests(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                raw = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            except (ValueError, OSError):
                break
            if not raw:
                await self._request_queue.put(FrontendRequest(type="shutdown"))
                break
            payload = raw.decode("utf-8").strip()
            if not payload:
                continue
            try:
                req = FrontendRequest.model_validate_json(payload)
            except Exception as exc:
                await self.emit(BackendEvent(type="error", message=f"Invalid request: {exc}"))
                continue
            await self._request_queue.put(req)

    # ── request dispatch ──────────────────────────────────────

    async def _handle_request(self, req: FrontendRequest) -> None:
        handler = {
            "discover": self._handle_discover,
            "load_model": self._handle_load_model,
            "generate": self._handle_generate,
            "stop_generation": self._handle_stop_generation,
            "eda_scan": self._handle_eda_scan,
            "eda_stats": self._handle_eda_stats,
            "eda_sample": self._handle_eda_sample,
            "train_launch": self._handle_train_launch,
            "train_stop": self._handle_train_stop,
            "train_status": self._handle_train_status,
            "bench_run": self._handle_bench_run,
            "cancel": self._handle_cancel,
            "shutdown": self._handle_shutdown,
        }.get(req.type)
        if handler:
            await handler(req)
        else:
            await self.emit(BackendEvent(type="error", message=f"Unknown request type: {req.type}"))

    async def _handle_discover(self, req: FrontendRequest) -> None:
        """Discover available model checkpoints."""
        from .discovery import discover_models
        try:
            models = discover_models()
            await self.emit(BackendEvent(type="model_list", data=models))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"Discovery failed: {e}"))

    async def _handle_load_model(self, req: FrontendRequest) -> None:
        """Load model by type. Supports .pth checkpoints and HF-format model directories."""
        try:
            from .discovery import resolve_weight_path, infer_config_from_ckpt
            import torch

            model_type = req.model_type or "lm"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Resolve weight path
            weight = req.weight or ""
            weight_path = resolve_weight_path(model_type, weight)
            if weight_path is None:
                await self.emit(BackendEvent(type="error", message=f"Weight not found: {weight}"))
                return

            # ── HF-format model directory (contains config.json) ──
            if os.path.isdir(weight_path):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                dtype = torch.float16 if device == "cuda" else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    weight_path,
                    torch_dtype=dtype,
                ).to(device).eval()
                self._tokenizer = AutoTokenizer.from_pretrained(weight_path)

                await self.emit(BackendEvent(
                    type="model_loaded",
                    message=f"{model_type.upper()} model loaded from {weight_path}",
                ))
                return

            # ── Raw .pth checkpoint ──
            ckpt = torch.load(weight_path, map_location=device)

            if model_type == "lm":
                from src.lm.model import LMConfig, LMForCausalLM
                cfg = infer_config_from_ckpt(ckpt, "lm", req)
                self._model = LMForCausalLM(cfg)
                sd = ckpt
                self._model.load_state_dict(sd, strict=False)
                self._model = self._model.half().eval().to(device)
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

            elif model_type == "vlm":
                from src.vlm.model_vlm import VLMConfig, VLM
                cfg = infer_config_from_ckpt(ckpt, "vlm", req)
                self._model = VLM(cfg, vision_model_path="model/siglip2-base-p32-256-ve")
                sd = {k: v for k, v in ckpt.items() if "mask" not in k}
                self._model.load_state_dict(sd, strict=False)
                self._model = self._model.half().eval().to(device)
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

            elif model_type == "rlm":
                from src.rlm.model import RLMConfig, RLM
                cfg = RLMConfig(dim=req.dim or 2048)
                self._model = RLM(cfg)
                sd = ckpt
                self._model.load_state_dict(sd, strict=False)
                self._model = self._model.half().eval().to(device)
                self._tokenizer = None

            await self.emit(BackendEvent(type="model_loaded", message=f"{model_type.upper()} model loaded from {weight_path}"))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"Load model failed: {e}"))

    async def _handle_generate(self, req: FrontendRequest) -> None:
        """Run generation, streaming tokens back."""
        if self._model is None:
            await self.emit(BackendEvent(type="error", message="No model loaded"))
            return
        try:
            import torch
            from transformers import TextStreamer
            from threading import Thread
            from queue import Queue

            tokenizer = self._tokenizer
            prompt = req.prompt or ""

            from src.rlm.model import RLM
            if isinstance(self._model, RLM):
                # RLM generate
                from src.rlm.tokenizer import RLMTokenizer
                tokenizer = RLMTokenizer()
                input_ids = tokenizer.encode(prompt).unsqueeze(0)
                output_ids = self._model.generate(input_ids, max_new_tokens=req.max_new_tokens or 512)
                text = tokenizer.decode(output_ids[0])
                await self.emit(BackendEvent(type="token", message=text))
                await self.emit(BackendEvent(type="done"))
                return

            # Apply chat template if the tokenizer has one (chat-tuned models)
            if tokenizer is not None and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # LM / VLM generate with streaming
            inputs = tokenizer(prompt, return_tensors="pt").to(next(self._model.parameters()).device)

            token_queue: Queue = Queue()

            class TuiStreamer(TextStreamer):
                def on_finalized_text(self, text: str, stream_end: bool = False):
                    token_queue.put(text)
                    if stream_end:
                        token_queue.put(None)

            streamer = TuiStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            def run_gen():
                with torch.no_grad():
                    self._model.generate(
                        inputs.input_ids,
                        max_new_tokens=req.max_new_tokens or 512,
                        do_sample=True,
                        temperature=req.temperature or 0.7,
                        top_p=req.top_p or 0.85,
                        streamer=streamer,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            thread = Thread(target=run_gen, daemon=True)
            thread.start()

            while True:
                text = await asyncio.get_event_loop().run_in_executor(None, token_queue.get)
                if text is None:
                    break
                await self.emit(BackendEvent(type="token", message=text))

            await self.emit(BackendEvent(type="done"))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"Generation failed: {e}"))

    async def _handle_stop_generation(self, req: FrontendRequest) -> None:
        await self.emit(BackendEvent(type="done", message="Generation stopped"))

    async def _handle_eda_scan(self, req: FrontendRequest) -> None:
        """Scan data_dir for .jsonl and .parquet files."""
        from .data_reader import scan_datasets
        try:
            result = scan_datasets(req.data_dir or "./datasets")
            await self.emit(BackendEvent(type="eda_result", data=result))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"EDA scan failed: {e}"))

    async def _handle_eda_stats(self, req: FrontendRequest) -> None:
        """Get stats for a specific dataset file."""
        from .data_reader import get_dataset_stats
        try:
            result = get_dataset_stats(req.file_path or "")
            await self.emit(BackendEvent(type="eda_stats_result", data=result))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"EDA stats failed: {e}"))

    async def _handle_eda_sample(self, req: FrontendRequest) -> None:
        """Get a sample from a dataset file."""
        from .data_reader import preview_sample
        try:
            result = preview_sample(req.file_path or "", req.index or 0)
            await self.emit(BackendEvent(type="eda_sample_result", data=result))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"EDA sample failed: {e}"))

    async def _handle_train_launch(self, req: FrontendRequest) -> None:
        """Launch training as subprocess."""
        from .train_launcher import launch_training
        try:
            result = await launch_training(req.train_cmd or [])
            await self.emit(BackendEvent(type="train_status", data=result))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"Train launch failed: {e}"))

    async def _handle_train_stop(self, req: FrontendRequest) -> None:
        if self._train_process:
            self._train_process.send_signal(signal.SIGTERM)
            await self.emit(BackendEvent(type="train_status", data={"status": "stopped"}))

    async def _handle_train_status(self, req: FrontendRequest) -> None:
        running = self._train_process is not None and self._train_process.returncode is None
        await self.emit(BackendEvent(type="train_status", data={"running": running}))

    async def _handle_bench_run(self, req: FrontendRequest) -> None:
        from .bench_runner import run_benchmark
        try:
            result = await run_benchmark(
                model_type=req.model_type or "lm",
                weight=req.weight or "",
                bench_name=req.bench_name or "perplexity",
            )
            await self.emit(BackendEvent(type="bench_result", data=result))
        except Exception as e:
            await self.emit(BackendEvent(type="error", message=f"Bench failed: {e}"))

    async def _handle_cancel(self, req: FrontendRequest) -> None:
        await self.emit(BackendEvent(type="done", message="Cancelled"))

    async def _handle_shutdown(self, req: FrontendRequest) -> None:
        self._running = False

    # ── main loop ──────────────────────────────────────────────

    async def run(self) -> None:
        # Emit ready + send discovered models
        from .discovery import discover_models
        models = discover_models()
        await self.emit(BackendEvent(
            type="ready",
            data={"model_types": ["lm", "vlm", "rlm"]},
        ))
        await self.emit(BackendEvent(
            type="model_list",
            data=models,
        ))
        reader_task = asyncio.create_task(self._read_requests())
        try:
            while self._running:
                try:
                    req = await asyncio.wait_for(self._request_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await self._handle_request(req)
        finally:
            reader_task.cancel()
            await self.emit(BackendEvent(type="shutdown"))
