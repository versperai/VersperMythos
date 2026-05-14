"""Dataset scanning and statistics for TUI EDA."""

import json
import os
from pathlib import Path
from typing import Any


def scan_datasets(data_dir: str = "./datasets") -> list[dict[str, Any]]:
    """Scan directory for .jsonl and .parquet files."""
    results = []
    base = Path(data_dir)
    if not base.exists():
        return results
    for p in base.rglob("*"):
        if p.suffix in (".jsonl", ".parquet"):
            rel = p.relative_to(base)
            ftype = "jsonl" if p.suffix == ".jsonl" else "parquet"
            size_mb = p.stat().st_size / (1024 * 1024)
            results.append({
                "path": str(p),
                "name": str(rel),
                "type": ftype,
                "size_mb": round(size_mb, 2),
            })
    return results


def get_dataset_stats(file_path: str) -> dict[str, Any]:
    """Get row count and basic stats for a dataset file."""
    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}

    if p.suffix == ".jsonl":
        return _stats_jsonl(p)
    elif p.suffix == ".parquet":
        return _stats_parquet(p)
    return {"error": f"Unsupported format: {p.suffix}"}


def _stats_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        return {"rows": 0}
    keys = list(rows[0].keys()) if rows else []
    # token length estimation
    text_lens = []
    for r in rows:
        if "text" in r:
            text_lens.append(len(r["text"]))
        elif "conversations" in r:
            text_lens.append(sum(len(c.get("content", "")) for c in r["conversations"]))
        else:
            text_lens.append(0)
    avg_len = sum(text_lens) / n if n else 0
    return {
        "rows": n,
        "keys": keys,
        "avg_text_length": round(avg_len, 1),
        "max_text_length": max(text_lens) if text_lens else 0,
        "min_text_length": min(text_lens) if text_lens else 0,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "format": "jsonl",
    }


def _stats_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        cols = [f.name for f in pf.schema]
        return {
            "rows": n,
            "columns": cols,
            "num_columns": len(cols),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "format": "parquet",
        }
    except Exception as e:
        return {"error": str(e)}


def preview_sample(file_path: str, index: int = 0) -> dict[str, Any]:
    """Get a single sample from a dataset."""
    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}

    if p.suffix == ".jsonl":
        with open(p) as f:
            for i, line in enumerate(f):
                if i == index and line.strip():
                    sample = json.loads(line)
                    # Truncate long text for display
                    for k, v in sample.items():
                        if isinstance(v, str) and len(v) > 500:
                            sample[k] = v[:500] + "..."
                    return {"index": index, "data": sample}
        return {"error": f"Index {index} out of range"}

    elif p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(p, offset=index, num_rows=1)
            sample = table.to_pydict()
            sample = {k: str(v[0])[:500] for k, v in sample.items()}
            return {"index": index, "data": sample}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unsupported format: {p.suffix}"}
