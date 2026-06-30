"""
Native PyTorch dataset for multi-label symptom classification.

Reads the already-normalized splits produced by training/preprocess.py. This
module imports nothing from the FastAPI app: training is fully standalone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TEXT_COL = "text"


class SymptomDataset(Dataset):
    """Tokenizes Arabic text lazily and returns tensors for the HF Trainer.

    Each item is a dict with `input_ids`, `attention_mask`, and `labels`
    (a float vector, as required by BCEWithLogitsLoss for multi-label).
    """

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 64):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype="float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def load_labels(labels_path: Path) -> List[str]:
    """Load the canonical, ordered label list (index -> symptom)."""
    return json.loads(Path(labels_path).read_text(encoding="utf-8"))


def load_split(csv_path: Path, label_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    """Load a split CSV into (texts, label matrix) using a fixed label order."""
    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[label_cols].to_numpy(dtype="float32")
    return texts, labels
