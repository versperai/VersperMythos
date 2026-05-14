"""Model checkpoint discovery for TUI."""

import json
from pathlib import Path
from typing import Any


def _infer_model_type(parent_dir: str, name: str) -> str:
    """Infer model type (lm/vlm/rlm) from parent directory or name."""
    if parent_dir == "vlm" or "vlm" in name:
        return "vlm"
    if parent_dir == "rlm" or "rlm" in name:
        return "rlm"
    return "lm"


def discover_models() -> list[dict[str, Any]]:
    """Scan out/ directories for .pth checkpoints and HF-format model directories."""
    results: list[dict[str, Any]] = []
    base = Path("out")
    if not base.exists():
        return results

    # ── 1. Raw .pth checkpoint files ──
    for p in base.rglob("*.pth"):
        size_mb = p.stat().st_size / (1024 * 1024)
        rel = str(p.relative_to(base))
        parent = p.parent.name
        name = p.stem
        model_type = _infer_model_type(parent, name)

        hidden_size = 768
        for part in name.split("_"):
            if part.isdigit():
                hidden_size = int(part)

        results.append({
            "name": rel,
            "path": str(p),
            "type": model_type,
            "hidden_size": hidden_size,
            "size_mb": round(size_mb, 2),
        })

    # ── 2. HF-format model directories (contain config.json with architecture info) ──
    seen: set[str] = set()
    for config_path in base.rglob("config.json"):
        hf_dir = config_path.parent
        if hf_dir == base:
            continue
        # Skip hidden directories (e.g. .ipynb_checkpoints)
        if any(part.startswith(".") for part in hf_dir.relative_to(base).parts):
            continue
        # Skip tokenizer configs
        if "tokenizer" in hf_dir.name.lower():
            continue

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception:
            continue

        # Must have model architecture info
        if "architectures" not in cfg and "model_type" not in cfg:
            continue

        dir_key = str(hf_dir.resolve())
        if dir_key in seen:
            continue
        seen.add(dir_key)

        size_mb = sum(
            f.stat().st_size for f in hf_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        rel = str(hf_dir.relative_to(base))
        parent = hf_dir.parent.name
        model_type = _infer_model_type(parent, hf_dir.name)
        hidden_size = cfg.get("hidden_size", cfg.get("dim", 768))

        results.append({
            "name": rel,
            "path": str(hf_dir),
            "type": model_type,
            "hidden_size": hidden_size,
            "size_mb": round(size_mb, 2),
        })

    return results


def resolve_weight_path(model_type: str, weight: str) -> str | None:
    """Find a weight file or HF model directory given model type and weight name hint.

    Tries, in order:
      1. Direct path (file or directory, as-is)
      2. out/{weight}/  —  HF directory (e.g. weight="lm/sft" → out/lm/sft/)
      3. out/{model_type}/{weight}/  —  HF directory
      4. out/{model_type}/{weight}_*.pth
      5. out/{weight}.pth or out/{weight}_*.pth
      6. out/**/{weight}*.pth
    """
    base = Path("out")
    if not base.exists():
        return None

    # Direct path (absolute or relative)
    p = Path(weight)
    if p.exists():
        return str(p)

    # HF directory: weight as relative path (e.g. "lm/sft" → out/lm/sft/)
    hf_dir = base / weight
    if hf_dir.is_dir() and (hf_dir / "config.json").exists():
        return str(hf_dir)

    # HF directory: out/{model_type}/{weight}/
    hf_dir = base / model_type / weight
    if hf_dir.is_dir() and (hf_dir / "config.json").exists():
        return str(hf_dir)

    # out/{model_type}/{weight}_*.pth
    candidates = list(base.glob(f"{model_type}/{weight}*.pth"))
    if candidates:
        return str(candidates[0])

    # out/{weight}.pth or out/{weight}_*.pth
    candidates = list(base.glob(f"{weight}.pth")) or list(base.glob(f"{weight}_*.pth"))
    if candidates:
        return str(candidates[0])

    # Recursive search
    candidates = list(base.rglob(f"**/{weight}*.pth"))
    if candidates:
        return str(candidates[0])

    return None


def infer_config_from_ckpt(
    ckpt: dict[str, Any],
    model_type: str,
    req: Any,  # FrontendRequest duck-type
) -> Any:  # returns config object matching model_type
    """Infer model config from checkpoint state dict keys.

    Sniffs architecture dimensions from weight tensor shapes so the
    caller doesn't need to specify num_key_value_heads, intermediate_size, etc.
    """
    import torch

    # Collect all layer keys to infer num_hidden_layers
    layer_keys = {k for k in ckpt if k.startswith("model.layers.")}
    n_layers = max(
        (int(k.split(".")[2]) for k in layer_keys if k.split(".")[2].isdigit()),
        default=8,
    ) + 1

    # Sniff q/k/v shapes from first layer
    q_key = next((k for k in ckpt if "self_attn.q_proj.weight" in k), None)
    k_key = next((k for k in ckpt if "self_attn.k_proj.weight" in k), None)

    hidden_size = ckpt[q_key].shape[1] if q_key else (req.hidden_size or 768)
    n_heads = (ckpt[q_key].shape[0] // 128) if q_key and ckpt[q_key].shape[0] % 128 == 0 else (req.num_attention_heads or 8)
    # head_dim = hidden_size / n_heads, but we don't have it directly — infer from shapes
    # q_proj is [n_heads * head_dim, hidden_size], k_proj is [n_kv_heads * head_dim, hidden_size]
    head_dim = hidden_size // n_heads
    n_kv_heads = (ckpt[k_key].shape[0] // head_dim) if k_key and head_dim > 0 else (req.num_key_value_heads or (n_heads if n_heads <= 4 else 4))

    # Sniff intermediate_size from MLP gate_proj
    gate_key = next((k for k in ckpt if "mlp.gate_proj.weight" in k), None)
    intermediate_size = ckpt[gate_key].shape[0] if gate_key else (req.intermediate_size or 2048)

    # Check for MoE
    use_moe = any("mlp.experts" in k for k in ckpt)

    if model_type == "lm":
        from src.lm.model import LMConfig
        return LMConfig(
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            use_moe=use_moe,
        )
    elif model_type == "vlm":
        from src.vlm.model_vlm import VLMConfig
        return VLMConfig(
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            use_moe=use_moe,
        )

    return None
