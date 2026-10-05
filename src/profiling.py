"""Measure runtime and peak memory of heavy steps and append them to the performance log.
"""
import csv
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
import pyarrow as pa
from src.config import PERFORMANCE_LOG

MB = 1024 ** 2
LOG_COLUMNS = ["timestamp", "step", "runtime_s", "peak_python_mb", "peak_arrow_mb", "details"]

@contextmanager
def track(step, **extra):
    """Time a block, record its peak memory, print it and log it; yields a stats dict."""
    stats = {"step": step, **extra}
    started_here = not tracemalloc.is_tracing()  # nested track() calls share one tracer
    if started_here:
        tracemalloc.start()
    tracemalloc.reset_peak()
    arrow_pool = pa.default_memory_pool()
    arrow_before = arrow_pool.bytes_allocated()

    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        stats["runtime_s"] = round(time.perf_counter() - t0, 3)
        stats["peak_python_mb"] = round(tracemalloc.get_traced_memory()[1] / MB, 1)
        # max_memory() is the pool's all-time peak, so this is an upper bound.
        stats["peak_arrow_mb"] = round(max(arrow_pool.max_memory() - arrow_before, 0) / MB, 1)
        if started_here:
            tracemalloc.stop()
        print(f"[{step}] runtime: {stats['runtime_s']:.2f} s | "
              f"peak memory (Python/pandas): {stats['peak_python_mb']:.1f} MB | "
              f"peak memory (pyarrow pool): {stats['peak_arrow_mb']:.1f} MB")
        _append_to_log(stats)


def _append_to_log(stats):
    fixed = ["step", "runtime_s", "peak_python_mb", "peak_arrow_mb"]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **{k: stats[k] for k in fixed},
        "details": "; ".join(f"{k}={v}" for k, v in stats.items() if k not in fixed),
    }
    PERFORMANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PERFORMANCE_LOG.exists()
    with open(PERFORMANCE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
