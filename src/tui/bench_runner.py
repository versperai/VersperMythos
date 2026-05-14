"""Benchmark runner for TUI."""

from typing import Any


async def run_benchmark(
    model_type: str = "lm",
    weight: str = "",
    bench_name: str = "perplexity",
) -> dict[str, Any]:
    """Run a benchmark and return results."""
    return {
        "benchmark": bench_name,
        "model_type": model_type,
        "weight": weight,
        "status": "not_implemented",
    }
