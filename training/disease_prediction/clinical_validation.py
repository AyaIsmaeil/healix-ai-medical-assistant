"""
Healix - Phase 6.2 clinical validation vs the DDXPlus differential (OFFLINE).

Standard classification metrics ask "was the single top prediction right?".
Clinically, that is the wrong question: a physician works from a DIFFERENTIAL
— a ranked shortlist. DDXPlus ships exactly that (``y_differential_ids`` /
``y_differential_probs``, mean length ~9.2 per the Phase-5 statistics), which
lets us evaluate the model the way a clinician would actually use it.

Metrics
-------
* Top-1 / Top-3 / Top-5 accuracy against the ground-truth pathology.
* MRR (Mean Reciprocal Rank) of the true disease in the model's ranking —
  rewards putting the truth high even when it is not first.
* Differential coverage - of the diseases DDXPlus lists in the reference
  differential, how many appear in the model's own top-k? This measures
  whether the model reproduces the *clinical shortlist*, not just the answer.
* Differential rank agreement - does the model's #1 match DDXPlus's #1?

Note on label provenance (from the Phase-5 ``label_dictionary.json``):
``y_differential`` is a STRONG label; ``y_specialty`` is degenerate and
``y_urgency_prior`` is a disease-level prior. Only the differential is used
here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _rank_of_truth(proba_row: np.ndarray, true_idx: int) -> int:
    """1-based rank of the true class in a descending probability ranking."""
    order = np.argsort(proba_row)[::-1]
    return int(np.flatnonzero(order == true_idx)[0]) + 1


def clinical_metrics(proba: np.ndarray, y_true: np.ndarray,
                     ks=(1, 3, 5)) -> Dict[str, Any]:
    n = len(y_true)
    ranks = np.array([_rank_of_truth(proba[i], int(y_true[i]))
                      for i in range(n)], dtype=int)

    out: Dict[str, Any] = {"n_samples": int(n)}
    for k in ks:
        out[f"top_{k}_accuracy"] = round(float(np.mean(ranks <= k)), 6)
    out["mrr"] = round(float(np.mean(1.0 / ranks)), 6)
    out["mean_rank_of_truth"] = round(float(ranks.mean()), 4)
    out["median_rank_of_truth"] = float(np.median(ranks))
    out["worst_rank_of_truth"] = int(ranks.max())
    # how often the truth falls outside a realistic clinical shortlist
    out["truth_outside_top5_pct"] = round(float(np.mean(ranks > 5) * 100), 4)
    return out


def differential_agreement(proba: np.ndarray, y_true: np.ndarray,
                           differential_ids: List[List[str]],
                           class_names: List[str],
                           k: int = 5) -> Dict[str, Any]:
    """Compare the model's top-k against the DDXPlus reference differential."""
    index_by_id = {cid: i for i, cid in enumerate(class_names)}
    n = len(y_true)

    coverages: List[float] = []
    top1_matches = 0
    n_with_diff = 0
    ref_lengths: List[int] = []

    for i in range(n):
        ref = differential_ids[i]
        if ref is None or len(ref) == 0:
            continue
        n_with_diff += 1
        ref_lengths.append(len(ref))

        model_topk = np.argsort(proba[i])[::-1][:k]
        model_topk_ids = {class_names[j] for j in model_topk}

        ref_ids = {r for r in ref if r in index_by_id}
        if ref_ids:
            coverages.append(len(ref_ids & model_topk_ids) / len(ref_ids))

        # does the model's #1 equal the differential's #1?
        if ref[0] in index_by_id and int(np.argmax(proba[i])) == index_by_id[ref[0]]:
            top1_matches += 1

    return {
        "k": k,
        "n_rows_with_reference_differential": int(n_with_diff),
        "mean_reference_differential_length": round(
            float(np.mean(ref_lengths)), 4) if ref_lengths else None,
        # fraction of the reference differential recovered inside model top-k
        "mean_differential_coverage_at_k": round(
            float(np.mean(coverages)), 6) if coverages else None,
        "model_top1_equals_reference_top1_pct": round(
            100 * top1_matches / n_with_diff, 4) if n_with_diff else None,
    }


def plot_rank_distribution(proba: np.ndarray, y_true: np.ndarray, title: str,
                           path, max_rank: int = 10) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    try:
        ranks = np.array([_rank_of_truth(proba[i], int(y_true[i]))
                          for i in range(len(y_true))])
        counts = [int(np.sum(ranks == r)) for r in range(1, max_rank + 1)]
        counts.append(int(np.sum(ranks > max_rank)))
        labels = [str(r) for r in range(1, max_rank + 1)] + [f">{max_rank}"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, counts, color="#2b8cbe")
        ax.set_yscale("log")
        ax.set_xlabel("rank of the TRUE disease in the model's ranking")
        ax.set_ylabel("patients (log scale)")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
