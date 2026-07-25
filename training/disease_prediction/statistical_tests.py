"""
Healix - Phase 6.2 statistical comparison of cross-validated models (OFFLINE).

Phase 6.1 ranked five families by a single validation split and honestly
flagged the top-2 gap (0.99615 vs 0.99594 = 0.0002) as "within plausible
single-split noise ... a statistical tie". This module replaces that informal
judgement with explicit hypothesis testing on the k-fold results.

THE METHODOLOGICAL PROBLEM (and why a naive t-test would be wrong)
------------------------------------------------------------------
The standard paired t-test assumes the k paired differences are INDEPENDENT.
In k-fold cross-validation they are not: any two folds share (k-2)/(k-1) of
their training data, so the fold scores are positively correlated. Feeding
them to an uncorrected t-test **understates the variance and inflates
significance** -- it is a well-documented way to manufacture false positives
(Dietterich 1998; Nadeau & Bengio 2003).

We therefore report three tests side by side and let the disagreement between
them be part of the evidence:

1. **Uncorrected paired t-test** - reported ONLY as a reference point, and
   explicitly labelled as anti-conservative. Never used alone to conclude.
2. **Nadeau-Bengio corrected resampled t-test** - inflates the variance by
   (1/k + n_test/n_train) to account for the train-set overlap. This is the
   primary test.
3. **Wilcoxon signed-rank** - distribution-free fallback. With k=5 its minimum
   attainable two-sided p-value is 0.0625, so it CANNOT reach p<0.05 no matter
   how large the effect. That is a property of the sample size, not evidence of
   absence, and is reported as such.

Also reported: 95% confidence intervals (t-distribution), paired Cohen's d,
and a comparison against a pre-declared practical-significance threshold --
because at a ceiling of ~0.996 a statistically detectable difference can still
be clinically meaningless.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------
# Pre-declared practical-significance threshold
# ----------------------------------------------------------------------
# Declared BEFORE looking at any Phase-6.2 result, so it cannot be tuned to
# produce a desired verdict.
#
# Rationale: the DDXPlus test split holds 134,529 patients. A macro-F1
# difference of 0.002 corresponds to roughly a few hundred patients' worth of
# changed per-class behaviour -- small enough that it is dominated by the known
# ~1.4% residual leakage and by the synthetic-vs-real gap documented in
# SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md. Anything below this is
# treated as practically negligible even if statistically detectable.
PRACTICAL_SIGNIFICANCE_DELTA = 0.002
ALPHA = 0.05


def descriptive_stats(values: Sequence[float], confidence: float = 0.95
                      ) -> Dict[str, Any]:
    """Mean, std, and a t-based confidence interval for a fold-score vector."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": round(mean, 6), "std": None,
                "ci_low": None, "ci_high": None, "sem": None}

    std = float(arr.std(ddof=1))
    sem = std / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "sem": round(sem, 6),
        "ci_level": confidence,
        "ci_low": round(mean - tcrit * sem, 6),
        "ci_high": round(mean + tcrit * sem, 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def _cohens_d_paired(diff: np.ndarray) -> Optional[float]:
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return None
    return float(diff.mean() / sd)


def compare_two_models(scores_a: Sequence[float], scores_b: Sequence[float],
                       name_a: str, name_b: str, metric: str = "f1_macro",
                       n_folds: Optional[int] = None,
                       alpha: float = ALPHA,
                       practical_delta: float = PRACTICAL_SIGNIFICANCE_DELTA
                       ) -> Dict[str, Any]:
    """Full paired statistical comparison of two models over identical folds.

    ``scores_a``/``scores_b`` MUST come from the same StratifiedKFold seed, so
    element i of each is the same held-out fold -- otherwise the pairing (and
    every test below) is invalid.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires equal-length fold vectors")

    k = n_folds or len(a)
    diff = a - b
    n = len(diff)

    out: Dict[str, Any] = {
        "metric": metric,
        "model_a": name_a,
        "model_b": name_b,
        "n_folds": n,
        "fold_scores_a": [round(float(v), 6) for v in a],
        "fold_scores_b": [round(float(v), 6) for v in b],
        "fold_differences": [round(float(v), 6) for v in diff],
        "descriptive_a": descriptive_stats(a),
        "descriptive_b": descriptive_stats(b),
        "mean_difference": round(float(diff.mean()), 6),
        "folds_a_wins": int((diff > 0).sum()),
        "folds_b_wins": int((diff < 0).sum()),
        "folds_tied": int((diff == 0).sum()),
    }

    if n < 2 or np.allclose(diff, 0):
        out["verdict"] = ("identical fold scores - no difference to test"
                          if np.allclose(diff, 0) else "too few folds to test")
        out["statistically_significant"] = False
        out["practically_significant"] = False
        return out

    sd_diff = float(diff.std(ddof=1))
    sem_diff = sd_diff / math.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    out["std_difference"] = round(sd_diff, 6)
    out["ci_difference_low"] = round(float(diff.mean()) - tcrit * sem_diff, 6)
    out["ci_difference_high"] = round(float(diff.mean()) + tcrit * sem_diff, 6)
    out["cohens_d_paired"] = (round(_cohens_d_paired(diff), 4)
                              if _cohens_d_paired(diff) is not None else None)

    # --- 1. uncorrected paired t-test (reference only; anti-conservative) ---
    t_unc, p_unc = stats.ttest_rel(a, b)
    out["paired_t_test_uncorrected"] = {
        "t": round(float(t_unc), 6), "p_value": round(float(p_unc), 6),
        "significant_at_alpha": bool(p_unc < alpha),
        "WARNING": ("assumes independent folds, which k-fold CV violates; "
                    "anti-conservative -- reported for reference only"),
    }

    # --- 2. Nadeau-Bengio corrected resampled t-test (PRIMARY) ---
    # variance inflation for k-fold: (1/k + n_test/n_train) with
    # n_test/n_train = 1/(k-1)
    correction = (1.0 / k) + (1.0 / (k - 1)) if k > 1 else 1.0
    denom = math.sqrt(correction * (sd_diff ** 2)) if sd_diff > 0 else 0.0
    if denom > 0:
        t_corr = float(diff.mean()) / denom
        p_corr = float(2 * (1 - stats.t.cdf(abs(t_corr), df=n - 1)))
    else:
        t_corr, p_corr = float("nan"), 1.0
    out["nadeau_bengio_corrected_t_test"] = {
        "t": round(t_corr, 6) if not math.isnan(t_corr) else None,
        "p_value": round(p_corr, 6),
        "significant_at_alpha": bool(p_corr < alpha),
        "variance_inflation_factor": round(correction, 6),
        "note": ("PRIMARY test: corrects for the training-set overlap between "
                 "folds (Nadeau & Bengio 2003)"),
    }

    # --- 3. Wilcoxon signed-rank (non-parametric) ---
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
        min_p = 2.0 ** (-(n - 1))   # smallest attainable two-sided p for n pairs
        out["wilcoxon_signed_rank"] = {
            "statistic": round(float(w_stat), 6),
            "p_value": round(float(p_w), 6),
            "significant_at_alpha": bool(p_w < alpha),
            "minimum_attainable_p_value": round(min_p, 6),
            "underpowered": bool(min_p >= alpha),
            "note": (f"with n={n} paired folds the smallest possible two-sided "
                     f"p is {min_p:.4f}; if that exceeds alpha the test CANNOT "
                     f"reach significance regardless of effect size"),
        }
    except Exception as exc:  # noqa: BLE001 - never break a report over a test
        out["wilcoxon_signed_rank"] = {"error": str(exc)}

    # --- verdicts ---
    stat_sig = bool(p_corr < alpha)
    prac_sig = bool(abs(float(diff.mean())) >= practical_delta)
    out["alpha"] = alpha
    out["practical_significance_delta"] = practical_delta
    out["statistically_significant"] = stat_sig
    out["practically_significant"] = prac_sig

    if stat_sig and prac_sig:
        verdict = (f"{name_a} is both statistically and practically better than "
                   f"{name_b} on {metric}")
    elif stat_sig and not prac_sig:
        verdict = (f"difference is statistically detectable but PRACTICALLY "
                   f"NEGLIGIBLE (|{diff.mean():.6f}| < {practical_delta}) - "
                   f"treat {name_a} and {name_b} as equivalent and decide on "
                   f"other criteria (cost, calibration, robustness)")
    elif not stat_sig and prac_sig:
        verdict = (f"observed gap exceeds the practical threshold but is NOT "
                   f"statistically significant at alpha={alpha} with {n} folds "
                   f"- underpowered; more folds/repeats needed to conclude")
    else:
        verdict = (f"NO significant difference and the gap is practically "
                   f"negligible - {name_a} and {name_b} are equivalent on {metric}")
    out["verdict"] = verdict
    return out


def compare_all(cv_results: Dict[str, Any], metric: str = "f1_macro",
                top_n: int = 3) -> Dict[str, Any]:
    """Pairwise-compare the top-N families from a cross_validation.json payload.

    Applies a Holm-Bonferroni correction across the pairwise family
    comparisons, because testing several pairs on the same folds inflates the
    family-wise error rate.
    """
    usable = {f: r for f, r in cv_results.items()
              if isinstance(r, dict) and "folds" in r}
    if len(usable) < 2:
        return {"error": "need at least two cross-validated families"}

    ranked = sorted(usable.items(),
                    key=lambda kv: -float(np.mean([f[metric] for f in kv[1]["folds"]])))
    selected = ranked[:top_n]

    per_family = {
        name: {
            "params_used": res.get("params_used"),
            **{m: descriptive_stats([f[m] for f in res["folds"]])
               for m in ("accuracy", "f1_macro", "f1_weighted",
                         "fit_seconds", "predict_seconds_per_1000",
                         "peak_memory_mb")
               if m in res["folds"][0]},
        }
        for name, res in usable.items()
    }

    comparisons: List[Dict[str, Any]] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a_name, a_res = selected[i]
            b_name, b_res = selected[j]
            comparisons.append(compare_two_models(
                [f[metric] for f in a_res["folds"]],
                [f[metric] for f in b_res["folds"]],
                a_name, b_name, metric=metric))

    # Holm-Bonferroni over the primary (corrected) p-values
    if comparisons:
        idx = sorted(range(len(comparisons)),
                     key=lambda i: comparisons[i]["nadeau_bengio_corrected_t_test"]["p_value"])
        m = len(comparisons)
        prev = 0.0
        for rank, i in enumerate(idx):
            p = comparisons[i]["nadeau_bengio_corrected_t_test"]["p_value"]
            adj = min(1.0, max(prev, (m - rank) * p))
            prev = adj
            comparisons[i]["holm_bonferroni_adjusted_p"] = round(adj, 6)
            comparisons[i]["significant_after_multiple_comparison_correction"] = \
                bool(adj < ALPHA)

    return {
        "metric": metric,
        "alpha": ALPHA,
        "practical_significance_delta": PRACTICAL_SIGNIFICANCE_DELTA,
        "ranking": [{"rank": i + 1, "family": n,
                     "mean": round(float(np.mean([f[metric] for f in r["folds"]])), 6),
                     "std": round(float(np.std([f[metric] for f in r["folds"]], ddof=1)), 6)}
                    for i, (n, r) in enumerate(ranked)],
        "descriptive_by_family": per_family,
        "pairwise_comparisons_top_n": comparisons,
        "multiple_comparison_correction": "Holm-Bonferroni over the "
                                          "Nadeau-Bengio corrected p-values",
    }
