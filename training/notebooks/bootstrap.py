"""
Shared bootstrap for training/notebooks/01–07.

Every notebook starts with:

    from training.notebooks.bootstrap import bootstrap
    ctx = bootstrap()
    repo_root = ctx.repo_root
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or cwd) until ``app/main.py`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "app" / "main.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Healix repo root (expected app/main.py). "
        "Open the notebook from the repository or set cwd to the repo root."
    )


@dataclass(frozen=True)
class NotebookContext:
    repo_root: Path
    ddxplus_dir: Path
    raw_train_csv: Path
    raw_validate_csv: Path
    raw_test_csv: Path
    processed_dir: Path
    training_models_dir: Path
    runtime_models_dir: Path
    reports_dir: Path


def paths(repo_root: Path | None = None) -> NotebookContext:
    root = repo_root or find_repo_root()
    ddx = root / "app" / "data" / "raw" / "ddxplus"
    return NotebookContext(
        repo_root=root,
        ddxplus_dir=ddx,
        raw_train_csv=ddx / "train.csv",
        raw_validate_csv=ddx / "validate.csv",
        raw_test_csv=ddx / "test.csv",
        processed_dir=root / "training" / "data" / "processed",
        training_models_dir=root / "training" / "models",
        runtime_models_dir=root / "models",
        reports_dir=root / "training" / "reports",
    )


def require_ddxplus(ctx: NotebookContext | None = None) -> NotebookContext:
    ctx = ctx or paths()
    if not ctx.raw_train_csv.exists():
        raise FileNotFoundError(
            f"DDXPlus not found: {ctx.raw_train_csv}\n\n"
            "Download first (quick ~90 MB dev set):\n"
            "  python tools/download_ddxplus.py --quick\n\n"
            "Or full training set (~670 MB):\n"
            "  python tools/download_ddxplus.py --full\n"
        )
    return ctx


def bootstrap(seed: int = 42) -> NotebookContext:
    """Standard first cell: repo on sys.path, fixed seed, return paths."""
    ctx = require_ddxplus()
    if str(ctx.repo_root) not in sys.path:
        sys.path.insert(0, str(ctx.repo_root))

    import numpy as np

    np.random.seed(seed)

    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    except Exception:  # noqa: BLE001
        pass

    print(f"[bootstrap] repo_root = {ctx.repo_root}")
    print(f"[bootstrap] train.csv = {ctx.raw_train_csv} ({ctx.raw_train_csv.stat().st_size / 1e6:.1f} MB)")
    return ctx
