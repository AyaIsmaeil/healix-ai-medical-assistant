"""
Healix - Label Builder (Phase 5, Stage 4).

Builds the four target artifacts defined in
``docs/research/DATASET_BUILDER_DESIGN.md`` §6. Each label carries explicit
provenance so a downstream consumer cannot mistake a weak label for a strong
one:

  y_disease         - single-class HEALIX_DISEASE_* from PATHOLOGY. Strong.
  y_differential    - ranked [(disease, probability)] from DIFFERENTIAL_DIAGNOSIS.
                      Structurally identical to DiseasePredictionResult.
  y_urgency_prior   - DISEASE-LEVEL prior from the inverted DDXPlus severity.
                      NOT patient-level triage — two patients with the same
                      disease receive the same label regardless of how sick
                      they are. Real triage needs ESI-style acuity data.
  y_specialty       - derived deterministically FROM the disease. Adds no
                      independent information; a model trained on it can only
                      re-learn a lookup. AHD is the right source for a real
                      specialty model.

Confidence has **no** trainable target in DDXPlus (it needs a correctness
signal that only exists after a model is evaluated), so none is emitted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.ml.ontology_mapper import OntologyMapper

# Specialty assignments frozen in HEALIX_ONTOLOGY.md §3.2 (recommended level).
# Only influenza / pneumonia are confirmed by the live specialty_lookup.yaml.
SPECIALTY_BY_PATHOLOGY: Dict[str, str] = {
    "Anaphylaxis": "Emergency / Allergy",
    "Larygospasm": "ENT / Emergency",
    "Ebola": "Infectious Diseases",
    "Possible NSTEMI / STEMI": "Cardiology",
    "Acute pulmonary edema": "Cardiology / Pulmonology",
    "Spontaneous pneumothorax": "Pulmonology / Thoracic Surgery",
    "Boerhaave": "General Surgery",
    "Epiglottitis": "ENT",
    "Guillain-Barré syndrome": "Neurology",
    "Croup": "Pediatrics / ENT",
    "PSVT": "Cardiology",
    "Scombroid food poisoning": "Emergency / Family Medicine",
    "Myocarditis": "Cardiology",
    "Acute dystonic reactions": "Neurology",
    "Unstable angina": "Cardiology",
    "Stable angina": "Cardiology",
    "Pulmonary embolism": "Pulmonology / Cardiology",
    "Cluster headache": "Neurology",
    "Spontaneous rib fracture": "Orthopedics",
    "GERD": "Gastroenterology",
    "HIV (initial infection)": "Infectious Diseases",
    "Inguinal hernia": "General Surgery",
    "Myasthenia gravis": "Neurology",
    "Atrial fibrillation": "Cardiology",
    "Bronchiectasis": "Pulmonology",
    "Chagas": "Infectious Diseases",
    "Tuberculosis": "Pulmonology / Infectious Diseases",
    "Bronchospasm / acute asthma exacerbation": "Pulmonology",
    "Acute COPD exacerbation / infection": "Pulmonology",
    "Influenza": "Family Medicine",
    "Pneumonia": "Pulmonology",
    "Bronchiolitis": "Pediatrics / Pulmonology",
    "Pulmonary neoplasm": "Oncology / Pulmonology",
    "Pancreatic neoplasm": "Oncology / Gastroenterology",
    "Anemia": "Internal Medicine / Hematology",
    "Viral pharyngitis": "Family Medicine / ENT",
    "Whooping cough": "Pediatrics / Infectious Diseases",
    "Acute laryngitis": "ENT",
    "Allergic sinusitis": "ENT / Allergy",
    "Localized edema": "Internal Medicine",
    "SLE": "Rheumatology",
    "Acute otitis media": "ENT / Pediatrics",
    "Bronchitis": "Family Medicine / Pulmonology",
    "Acute rhinosinusitis": "ENT",
    "Sarcoidosis": "Pulmonology",
    "Pericarditis": "Cardiology",
    "Panic attack": "Psychiatry",
    "URTI": "Family Medicine",
    "Chronic rhinosinusitis": "ENT",
}


class LabelError(RuntimeError):
    """Raised when a label cannot be resolved to the frozen ontology."""


class LabelBuilder:
    """Builds the four aligned target artifacts for one row."""

    def __init__(self, mapper: OntologyMapper):
        self._m = mapper

    def build(self, pathology: str,
              differential: List[Tuple[str, float]]) -> Dict[str, Any]:
        pathology = (pathology or "").strip()
        disease = self._m.resolve_disease(pathology)

        specialty = SPECIALTY_BY_PATHOLOGY.get(pathology)
        if specialty is None:
            raise LabelError(f"No frozen specialty mapping for {pathology!r}")

        top_disease: Optional[str] = None
        top_prob: Optional[float] = None
        diff_ids: List[str] = []
        diff_probs: List[float] = []
        for name, prob in sorted(differential, key=lambda kv: -kv[1]):
            resolved = self._m.resolve_disease(name.strip())
            diff_ids.append(resolved["healix_id"])
            diff_probs.append(prob)
        if diff_ids:
            top_disease, top_prob = diff_ids[0], diff_probs[0]

        return {
            "y_disease": disease["healix_id"],
            "y_disease_icd10": disease["icd10"],
            "y_urgency_prior": disease["urgency_prior"],
            "y_specialty": specialty,
            "y_differential_ids": diff_ids,
            "y_differential_probs": diff_probs,
            "differential_length": len(diff_ids),
            "differential_top_id": top_disease,
            "differential_top_prob": top_prob,
            # QC signal: is the ground truth present in its own differential?
            "truth_in_differential": disease["healix_id"] in diff_ids,
        }

    @staticmethod
    def label_dictionary(mapper: OntologyMapper) -> Dict[str, Any]:
        """Machine-readable description of every label (label_dictionary.json)."""
        diseases = {
            meta["healix_id"]: {
                "pathology": name,
                "icd10": meta["icd10"],
                "ddxplus_severity": meta["severity"],
                "urgency_prior": meta["urgency_prior"],
                "specialty": SPECIALTY_BY_PATHOLOGY.get(name),
            }
            for name, meta in sorted(mapper.disease_by_pathology.items())
        }
        return {
            "ontology_version": "healix-ontology-v1.0.0",
            "targets": {
                "y_disease": {
                    "type": "multiclass", "classes": len(diseases),
                    "strength": "strong",
                    "description": "Ground-truth pathology as a Healix disease ID.",
                },
                "y_differential": {
                    "type": "ranked_multilabel",
                    "strength": "strong",
                    "description": "Ranked differential with probabilities; "
                                   "structurally identical to DiseasePredictionResult.",
                },
                "y_urgency_prior": {
                    "type": "ordinal_4",
                    "classes": ["EMERGENCY", "URGENT", "SEMI_URGENT", "NON_URGENT"],
                    "strength": "weak",
                    "warning": "DISEASE-LEVEL prior derived from the INVERTED DDXPlus "
                               "severity (1 = most urgent). Not patient-level triage.",
                },
                "y_specialty": {
                    "type": "multiclass",
                    "strength": "degenerate",
                    "warning": "Derived deterministically from y_disease; adds no "
                               "independent signal. Use AHD for a real specialty model.",
                },
                "y_confidence": {
                    "type": "none",
                    "strength": "unavailable",
                    "warning": "No trainable confidence target exists in DDXPlus.",
                },
            },
            "diseases": diseases,
        }
