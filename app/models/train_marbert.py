import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

MODEL_NAME = "UBC-NLP/MARBERT"
DATA_PATH = "data/train_samples.csv"
SYMPTOMS_PATH = "data/symptoms.json"
SAVE_DIR = "models/marabert_symptom_model"

with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

df = pd.read_csv(DATA_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_labels(symptom_string):
    vec = [0] * len(LABELS)
    if isinstance(symptom_string, str) and symptom_string.strip():
        symptoms = symptom_string.split("|")
        for s in symptoms:
            s = s.strip()
            if s in label2id:
                vec[label2id[s]] = 1
    return vec

class SymptomDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = [encode_labels(x) for x in dataframe["symptoms"].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label
        }

train_dataset = SymptomDataset(train_df)
val_dataset = SymptomDataset(val_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}

training_args = TrainingArguments(
    output_dir="models/output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

import os
os.makedirs(SAVE_DIR, exist_ok=True)

trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(f"{SAVE_DIR}/labels.json", "w", encoding="utf-8") as f:
    json.dump(LABELS, f, ensure_ascii=False, indent=2)

print("MARBERT model saved successfully.")