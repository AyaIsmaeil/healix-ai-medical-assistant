"""
Healix - Disease Prediction training utils (Phase 6.1).

Reproducibility, timing, and memory-sampling helpers shared by every script
in this package. Nothing here touches the runtime application.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every RNG this package touches, for reproducible baselines."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class Timer:
    """Context manager measuring wall-clock seconds. ``.elapsed`` after exit."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed: Optional[float] = None
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = round(time.perf_counter() - self._start, 4)


class PeakMemorySampler:
    """Samples this process's RSS in a background thread to approximate peak
    memory usage during a fit/predict call (a single before/after snapshot
    would miss the actual peak reached mid-computation)."""

    def __init__(self, interval_seconds: float = 0.2):
        self._interval = interval_seconds
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        import psutil
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = process.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:  # noqa: BLE001 - sampler must never crash training
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMemorySampler":
        import psutil
        self._peak_bytes = psutil.Process(os.getpid()).memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_bytes / 1e6, 1)


@contextmanager
def timed_block() -> Iterator[Timer]:
    t = Timer()
    with t:
        yield t


class NumpyJSONEncoder(json.JSONEncoder):
    """json.dumps helper — numpy scalars/arrays are common in metric dicts."""

    def default(self, obj: Any) -> Any:  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=NumpyJSONEncoder),
                    encoding="utf-8")


def directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
