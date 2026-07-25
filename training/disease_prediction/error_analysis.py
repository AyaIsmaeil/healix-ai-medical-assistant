"""
Healix - Phase 6.2 error analysis (OFFLINE ONLY).

Produces the confusion structure and, critically, EXPLAINS it using the
HEALIX ontology rather than just printing a matrix. For each top confusing
disease pair we pull the two conditions' DDXPlus symptom/antecedent sets and
report their Jaccard overlap, shared evidences, and distinguishing evidences
— which turns "the model confuses A and B" into "A and B share 82% of their
evidence set and differ only on X, Y".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return {class_names[i]: {"precision": round(float(precision[i]), 6),
                             "recall": round(float(recall[i]), 6),
                             "f1": round(float(f1[i]), 6),
                             "support": int(support[i])}
            for i in range(len(class_names))}


def top_confusions(cm: np.ndarray, class_names: List[str],
                   top_n: int = 20) -> List[Dict[str, Any]]:
    """Most frequent off-diagonal (true -> predicted) mistakes."""
    pairs: List[Tuple[int, int, int]] = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)

    out: List[Dict[str, Any]] = []
    for rank, (count, i, j) in enumerate(pairs[:top_n], start=1):
        support = int(cm[i].sum())
        out.append({
            "rank": rank,
            "true_id": class_names[i],
            "predicted_id": class_names[j],
            "count": count,
            "true_class_support": support,
            "pct_of_true_class": round(100 * count / support, 4) if support else None,
        })
    return out


def symmetric_confusion_pairs(cm: np.ndarray, class_names: List[str],
                              top_n: int = 15) -> List[Dict[str, Any]]:
    """Bidirectional confusion: A->B plus B->A. Surfaces genuinely
    indistinguishable PAIRS rather than one-directional bias."""
    n = cm.shape[0]
    scored: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            total = int(cm[i, j] + cm[j, i])
            if total > 0:
                scored.append((total, i, j))
    scored.sort(reverse=True)

    out = []
    for rank, (total, i, j) in enumerate(scored[:top_n], start=1):
        out.append({
            "rank": rank,
            "disease_a": class_names[i], "disease_b": class_names[j],
            "a_to_b": int(cm[i, j]), "b_to_a": int(cm[j, i]),
            "total_confusions": total,
            "support_a": int(cm[i].sum()), "support_b": int(cm[j].sum()),
        })
    return out


# ----------------------------------------------------------------------
# Ontology-grounded explanation of WHY two diseases get confused
# ----------------------------------------------------------------------
class ConfusionExplainer:
    """Explains a confused disease pair via their DDXPlus evidence sets."""

    def __init__(self, dataset_dir: Path, ddxplus_dir: Path):
        label_dict = json.loads(
            (dataset_dir / "metadata" / "label_dictionary.json")
            .read_text(encoding="utf-8"))
        self.diseases: Dict[str, Any] = label_dict["diseases"]

        self.conditions = json.loads(
            (ddxplus_dir / "release_conditions.json").read_text(encoding="utf-8"))
        evidences = json.loads(
            (ddxplus_dir / "release_evidences.json").read_text(encoding="utf-8"))
        self.evidence_text = {
            c: (spec.get("question_en") or "").strip()
            for c, spec in evidences.items()}

        # HEALIX_DISEASE_xxxx -> DDXPlus pathology name
        self.pathology_by_id = {did: meta["pathology"]
                                for did, meta in self.diseases.items()}

    def _evidence_set(self, healix_id: str) -> set:
        pathology = self.pathology_by_id.get(healix_id)
        cond = self.conditions.get(pathology, {})
        return set(cond.get("symptoms", {})) | set(cond.get("antecedents", {}))

    def explain_pair(self, id_a: str, id_b: str,
                     max_listed: int = 8) -> Dict[str, Any]:
        a, b = self._evidence_set(id_a), self._evidence_set(id_b)
        shared = a & b
        only_a, only_b = a - b, b - a
        union = a | b
        jaccard = len(shared) / len(union) if union else 0.0

        def describe(codes) -> List[str]:
            return [f"{c}: {self.evidence_text.get(c, '(no text)')}"
                    for c in sorted(codes)[:max_listed]]

        meta_a = self.diseases.get(id_a, {})
        meta_b = self.diseases.get(id_b, {})
        return {
            "disease_a": id_a, "pathology_a": self.pathology_by_id.get(id_a),
            "icd10_a": meta_a.get("icd10"), "specialty_a": meta_a.get("specialty"),
            "disease_b": id_b, "pathology_b": self.pathology_by_id.get(id_b),
            "icd10_b": meta_b.get("icd10"), "specialty_b": meta_b.get("specialty"),
            "same_specialty": meta_a.get("specialty") == meta_b.get("specialty"),
            "urgency_a": meta_a.get("urgency_prior"),
            "urgency_b": meta_b.get("urgency_prior"),
            # a confusion that crosses an urgency boundary is clinically worse
            "crosses_urgency_boundary":
                meta_a.get("urgency_prior") != meta_b.get("urgency_prior"),
            "n_evidences_a": len(a), "n_evidences_b": len(b),
            "n_shared_evidences": len(shared),
            "jaccard_overlap": round(jaccard, 4),
            "shared_evidences": describe(shared),
            "distinguishing_only_a": describe(only_a),
            "distinguishing_only_b": describe(only_b),
        }


def plot_confusion_matrix(cm: np.ndarray, title: str, path,
                          normalize: bool = True) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    try:
        m = cm.astype(float)
        if normalize:
            rows = m.sum(axis=1, keepdims=True)
            m = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1 if normalize else None)
        ax.set_title(title)
        ax.set_xlabel("predicted class index")
        ax.set_ylabel("true class index")
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="row-normalized rate" if normalize else "count")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
