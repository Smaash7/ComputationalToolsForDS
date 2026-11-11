"""Low-overhead wall-time and process-memory measurement."""

import threading
import time
from contextlib import contextmanager

import psutil


@contextmanager
def measure():
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def monitor():
        while not stop.wait(0.25):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
        result["peak_process_mb"] = peak[0] / (1024 ** 2)
