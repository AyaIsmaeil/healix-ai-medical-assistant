import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DISEASES_PATH = os.path.join(BASE_DIR, "database", "diseases.json")

with open(DISEASES_PATH, "r", encoding="utf-8") as f:
    DISEASES = json.load(f)


def diagnose(symptoms: list):

    if not symptoms:
        return {
            "disease": None,
            "confidence": 0,
            "message": "No symptoms detected"
        }

    scores = {}

    for disease, data in DISEASES.items():
        disease_symptoms = data["symptoms"]

        match = len(set(symptoms) & set(disease_symptoms))
        score = match / len(disease_symptoms)

        scores[disease] = score

    best = max(scores, key=scores.get)

    return {
        "disease": best,
        "confidence": round(scores[best], 2),
        "all_scores": scores
    }