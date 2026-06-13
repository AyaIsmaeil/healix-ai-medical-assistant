import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FILE_PATH = os.path.join(BASE_DIR, "database", "symptoms.json")

with open(FILE_PATH, "r", encoding="utf-8") as f:
    SYMPTOMS = json.load(f)


def extract_symptoms(text: str):
    text = text.lower()
    found = []

    for symptom, data in SYMPTOMS.items():

        # direct match (english key)
        if symptom in text:
            found.append(symptom)

        # syrian slang match
        for phrase in data["syrian"]:
            if phrase in text:
                found.append(symptom)

    return list(set(found))