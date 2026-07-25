"""
Healix - Phase 6.2 robustness under missing symptoms (OFFLINE ONLY).

Why this matters more than the headline accuracy: DDXPlus is a *complete*
simulated interview (closed-world — every unlisted evidence is a confirmed
negative, per the Phase-5 manifest's ``absence_semantics: closed_world``).
Real Healix conversations are open-world and truncated: the patient stops
talking, the LLM interview ends early, symptoms are simply never asked.

So the operative question is not "how accurate is the model on complete
records?" but "how fast does it degrade as the record becomes incomplete?".

Masking protocol
----------------
We hide a random subset of the SYMPTOM-presence columns only (the ``sym_``
block), leaving demographics/history/exposure intact, because that is what an
abbreviated interview actually loses.

Two masking semantics are measured separately, because they mean different
things clinically and the gap between them is itself a finding:

* ``to_nan``      - the symptom becomes UNKNOWN (never asked). This is what a
                    truncated Healix interview genuinely produces.
* ``to_zero``     - the symptom is asserted ABSENT. This is what a naive
                    pipeline does if it coerces missing to 0, and it injects
                    false negative evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from training.disease_prediction.config import SEED


def symptom_column_indices(feature_order: List[str]) -> np.ndarray:
    return np.array([i for i, c in enumerate(feature_order)
                     if c.startswith("sym_")], dtype=int)


def mask_features(X: np.ndarray, columns: np.ndarray, rate: float,
                  mode: str = "to_nan", seed: int = SEED) -> np.ndarray:
    """Return a copy of X with ``rate`` of ``columns`` hidden, per row.

    Masking is per-row (each patient loses a different random subset), which
    mirrors reality far better than dropping the same columns globally.
    """
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    n_rows = Xm.shape[0]
    n_mask = int(round(rate * len(columns)))
    if n_mask == 0:
        return Xm

    fill = np.nan if mode == "to_nan" else 0.0
    for r in range(n_rows):
        pick = rng.choice(columns, size=n_mask, replace=False)
        Xm[r, pick] = fill
    return Xm


def robustness_curve(model, X: np.ndarray, y: np.ndarray,
                     feature_order: List[str],
                     rates=(0.10, 0.20, 0.30, 0.40),
                     modes=("to_nan", "to_zero"),
                     seed: int = SEED,
                     logger=None) -> Dict[str, Any]:
    """Degradation curve across masking rates and semantics."""
    sym_cols = symptom_column_indices(feature_order)

    base_pred = model.predict(X)
    baseline = {
        "accuracy": round(float(accuracy_score(y, base_pred)), 6),
        "f1_macro": round(float(f1_score(y, base_pred, average="macro",
                                         zero_division=0)), 6),
    }
    if logger:
        logger.info("  robustness baseline acc=%.5f f1_macro=%.5f",
                    baseline["accuracy"], baseline["f1_macro"])

    curves: Dict[str, List[Dict[str, Any]]] = {}
    for mode in modes:
        rows: List[Dict[str, Any]] = []
        for rate in rates:
            Xm = mask_features(X, sym_cols, rate, mode=mode, seed=seed)
            pred = model.predict(Xm)
            acc = float(accuracy_score(y, pred))
            f1m = float(f1_score(y, pred, average="macro", zero_division=0))
            rec = {
                "missing_rate": rate,
                "n_symptom_columns_masked": int(round(rate * len(sym_cols))),
                "accuracy": round(acc, 6),
                "f1_macro": round(f1m, 6),
                "accuracy_drop": round(baseline["accuracy"] - acc, 6),
                "f1_macro_drop": round(baseline["f1_macro"] - f1m, 6),
                "f1_macro_retained_pct": round(
                    100 * f1m / baseline["f1_macro"], 3)
                if baseline["f1_macro"] else None,
            }
            rows.append(rec)
            if logger:
                logger.info("  [%s] rate=%.0f%% acc=%.5f f1_macro=%.5f "
                            "(retained %.1f%%)", mode, rate * 100, acc, f1m,
                            rec["f1_macro_retained_pct"] or 0.0)
        curves[mode] = rows

    return {
        "n_samples": int(len(y)),
        "n_symptom_columns": int(len(sym_cols)),
        "baseline": baseline,
        "curves": curves,
        "masking_semantics": {
            "to_nan": "symptom becomes UNKNOWN (never asked) — matches a "
                      "truncated real Healix interview",
            "to_zero": "symptom asserted ABSENT — what a naive missing->0 "
                       "pipeline injects; false negative evidence",
        },
    }


def plot_robustness(report: Dict[str, Any], title: str, path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        base = report["baseline"]["f1_macro"]
        for mode, rows in report["curves"].items():
            xs = [0.0] + [r["missing_rate"] * 100 for r in rows]
            ys = [base] + [r["f1_macro"] for r in rows]
            ax.plot(xs, ys, "o-", label=mode)
        ax.axhline(base, color="k", ls="--", lw=1, alpha=0.5,
                   label="complete-record baseline")
        ax.set_xlabel("% of symptom features hidden")
        ax.set_ylabel("macro-F1")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
