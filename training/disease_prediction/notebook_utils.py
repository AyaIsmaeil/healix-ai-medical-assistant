"""
Healix - shared notebook bootstrap (Phase 6, notebook-based workflow).

Every notebook under ``notebooks/`` starts with the same three lines:

    from training.disease_prediction.notebook_utils import bootstrap
    paths = bootstrap("07_probability_calibration")

This is the ONLY new logic introduced for the notebook methodology change --
it exists purely to remove ~15 lines of repo-root path wrangling and seeding
boilerplate from the top of every notebook (requirement: "keep reusable
utilities inside Python modules ... but the experiment execution itself must
live in notebooks"). All actual experiment logic continues to live in the
existing Phase 6.1/6.2 modules (``dataset.py``, ``models.py``, ``optimize.py``,
``cross_validation.py``, ``statistical_tests.py``, ``calibration.py``,
``explainability.py``, ``robustness.py``, ``error_analysis.py``,
``clinical_validation.py``, ``reporting.py``) -- notebooks import and
orchestrate them, they do not reimplement them.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class NotebookPaths:
    notebook_name: str
    repo_root: Path
    reports_dir: Path            # existing Phase 6.1/6.2 source-of-truth JSON
    models_dir: Path
    docs_research_dir: Path
    notebook_output_dir: Path    # THIS notebook's own outputs/<name>/ folder


def bootstrap(notebook_name: str, seed: int = 20260721) -> NotebookPaths:
    """Standard first cell for every notebook: repo-root on sys.path, fixed
    seeds for every RNG this project touches, consistent plot style, and a
    dedicated ``notebooks/outputs/<notebook_name>/`` directory for anything
    the notebook itself produces."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from training.disease_prediction.utils import set_global_seed
    set_global_seed(seed)

    try:
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
    except Exception:  # noqa: BLE001 - fall back silently outside Jupyter
        pass
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
            "font.size": 10,
        })
    except Exception:  # noqa: BLE001
        pass

    out_dir = REPO_ROOT / "notebooks" / "outputs" / notebook_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from training.disease_prediction import config as train_config
    print(f"[bootstrap] notebook={notebook_name!r}  seed={seed}")
    print(f"[bootstrap] repo_root         = {REPO_ROOT}")
    print(f"[bootstrap] reports (source)  = {train_config.REPORTS_DIR}")
    print(f"[bootstrap] notebook outputs  = {out_dir}")

    return NotebookPaths(
        notebook_name=notebook_name,
        repo_root=REPO_ROOT,
        reports_dir=train_config.REPORTS_DIR,
        models_dir=train_config.MODELS_DIR,
        docs_research_dir=train_config.DOCS_RESEARCH_DIR,
        notebook_output_dir=out_dir,
    )


def load_json(path: Path) -> Optional[Any]:
    """Load a JSON artifact, returning None (never a fabricated stand-in) if
    the experiment that would produce it has not run yet."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_json(path: Path, experiment_label: str) -> Any:
    """Like load_json, but raises a clear, actionable error instead of
    silently continuing -- used when a notebook's later cells cannot produce
    honest output without this artifact."""
    data = load_json(path)
    if data is None:
        raise FileNotFoundError(
            f"'{experiment_label}' has not been run yet -- {path} does not "
            f"exist. This notebook cell requires that experiment's real "
            f"output and will not fabricate a placeholder. Run the "
            f"corresponding upstream notebook/module first.")
    return data


def pending_banner(experiment_label: str, how_to_run: str) -> str:
    """Markdown text for a clearly-marked PENDING section (never invented
    results)."""
    return (
        f"### ⚠️ PENDING — {experiment_label} has not been executed\n\n"
        f"No fabricated numbers are shown here. To produce real results, "
        f"run:\n\n```\n{how_to_run}\n```\n\n"
        f"This section will be filled in by re-running this notebook after "
        f"that command completes."
    )
