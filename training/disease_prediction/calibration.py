"""
Healix - Phase 6.2 probability calibration analysis (OFFLINE ONLY).

Answers one clinical question: **can the model's predicted probabilities be
trusted as probabilities?** A disease model that is 99% accurate but whose
"0.8 confidence" really means 0.5 is dangerous downstream, because Healix's
Confidence Estimation stage is designed to consume these numbers literally.

Metrics
-------
* ECE  - Expected Calibration Error: |confidence - accuracy| averaged over
         confidence bins, weighted by bin population. 0 = perfect.
* MCE  - Maximum Calibration Error: the worst single bin. Surfaces a
         dangerous pocket that ECE's averaging can hide.
* Brier- multiclass Brier score (mean squared error over the one-hot target).
         Decomposes accuracy AND calibration together; lower is better.
* Reliability diagram - confidence vs. empirical accuracy per bin.

BEST_MODEL.md flagged that Random Forest's predict_proba outputs are raw
tree-vote fractions, NOT calibrated probabilities. This module measures
exactly how far off they are rather than assuming.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> Dict[str, Any]:
    """Top-label ECE/MCE with equal-width bins over [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts conf exactly 0 in bin 0 after clipping
    idx = np.clip(np.digitize(confidences, bins[1:-1], right=True), 0, n_bins - 1)

    rows: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    n = len(confidences)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": b, "lo": round(float(bins[b]), 4),
                         "hi": round(float(bins[b + 1]), 4), "count": 0,
                         "mean_confidence": None, "empirical_accuracy": None,
                         "gap": None})
            continue
        mean_conf = float(confidences[mask].mean())
        emp_acc = float(correct[mask].mean())
        gap = abs(mean_conf - emp_acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "lo": round(float(bins[b]), 4),
                     "hi": round(float(bins[b + 1]), 4), "count": count,
                     "mean_confidence": round(mean_conf, 6),
                     "empirical_accuracy": round(emp_acc, 6),
                     "gap": round(gap, 6)})

    return {"n_bins": n_bins, "ece": round(ece, 6), "mce": round(mce, 6),
            "n_samples": int(n), "bins": rows}


def multiclass_brier(proba: np.ndarray, y_true: np.ndarray,
                     n_classes: int) -> float:
    """Mean squared error between the full probability vector and one-hot
    truth. Uses the standard (non-doubled) definition: range [0, 2]."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibration_report(proba: np.ndarray, y_true: np.ndarray, n_classes: int,
                       n_bins: int = 15) -> Dict[str, Any]:
    """Full calibration assessment for one model on one sample."""
    pred = np.argmax(proba, axis=1)
    confidences = proba[np.arange(len(pred)), pred]
    correct = (pred == y_true).astype(float)

    ece = expected_calibration_error(confidences, correct, n_bins=n_bins)
    brier = multiclass_brier(proba, y_true, n_classes)

    overconf = float(np.mean(confidences - correct))
    return {
        "ece": ece["ece"], "mce": ece["mce"],
        "brier_multiclass": round(brier, 6),
        "mean_confidence": round(float(confidences.mean()), 6),
        "empirical_accuracy": round(float(correct.mean()), 6),
        # positive => over-confident, negative => under-confident
        "mean_confidence_minus_accuracy": round(overconf, 6),
        "direction": ("over-confident" if overconf > 0.005 else
                      "under-confident" if overconf < -0.005 else
                      "well-calibrated"),
        "reliability_bins": ece["bins"],
        "n_samples": int(len(y_true)),
    }


def plot_reliability_diagram(report: Dict[str, Any], title: str, path) -> Optional[str]:
    """Reliability diagram: mean confidence vs empirical accuracy per bin."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plots are best-effort
        return None

    try:
        bins = [b for b in report["reliability_bins"] if b["count"] > 0]
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["empirical_accuracy"] for b in bins]
        counts = [b["count"] for b in bins]

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#1f77b4", label="model")
        ax.set_xlabel("mean predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"{title}\nECE={report['ece']:.5f}  MCE={report['mce']:.5f}  "
                     f"Brier={report['brier_multiclass']:.5f}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        ax2.bar(range(len(counts)), counts, color="#888")
        ax2.set_yscale("log")
        ax2.set_xlabel("populated confidence bin (low -> high)")
        ax2.set_ylabel("samples (log)")
        ax2.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
