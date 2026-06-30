"""
Phase 1 - Data preprocessing for Arabic symptom extraction.

Reads the raw multi-label dataset (app/data/raw/symptoms.csv), cleans it, and
writes reproducible train/val/test splits plus the canonical label list to
app/data/processed/.

The raw file is treated as immutable; this script never writes to raw/.

Run from the repository root:
    python training/preprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make `app` importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.arabic import normalize_arabic

# --- Configuration -----------------------------------------------------------
RAW_PATH = REPO_ROOT / "app" / "data" / "raw" / "symptoms.csv"
OUT_DIR = REPO_ROOT / "app" / "data" / "processed"
TEXT_COL = "text"
# Labels with zero positive examples in symptoms.csv (cannot be learned).
DEAD_LABELS = ["hand_pain", "muscle_weakness", "foot_pain"]
SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def load_raw() -> pd.DataFrame:
    """Load the raw dataset, stripping the UTF-8 BOM on the `text` header."""
    return pd.read_csv(RAW_PATH, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": str(RAW_PATH.relative_to(REPO_ROOT))}

    df = load_raw()
    report["raw_rows"] = int(len(df))

    # 1. Drop dead labels, fixing the canonical label order (column order).
    all_labels = [c for c in df.columns if c != TEXT_COL]
    labels = [c for c in all_labels if c not in DEAD_LABELS]
    df = df[[TEXT_COL] + labels].copy()
    report["dropped_dead_labels"] = DEAD_LABELS
    report["num_labels"] = len(labels)

    # 2. Normalize text (identical to inference-time normalization).
    df[TEXT_COL] = df[TEXT_COL].map(normalize_arabic)

    # 3. Remove rows that became empty after normalization.
    before = len(df)
    df = df[df[TEXT_COL].str.len() > 0].copy()
    report["removed_empty_after_norm"] = int(before - len(df))

    # 4. Drop exact duplicate rows (same text AND same label vector). This must
    #    run AFTER normalization, which can create new duplicates.
    before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL] + labels).copy()
    report["removed_exact_duplicates"] = int(before - len(df))

    # 5. Detect conflicts: identical text carrying different label vectors.
    conflict_mask = df[TEXT_COL].duplicated(keep=False)
    report["conflicting_text_rows"] = int(conflict_mask.sum())

    report["clean_rows"] = int(len(df))

    # 6. Reproducible 80/10/10 split.
    holdout = VAL_FRACTION + TEST_FRACTION
    train_df, temp_df = train_test_split(
        df, test_size=holdout, random_state=SEED, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_FRACTION / holdout,
        random_state=SEED,
        shuffle=True,
    )

    # 7. Coverage check: every label must appear in every split.
    coverage: dict = {}
    missing: list = []
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        pos = part[labels].sum()
        coverage[name] = {k: int(v) for k, v in pos.items()}
        absent = [k for k, v in pos.items() if v == 0]
        if absent:
            missing.append({name: absent})

    report["split_rows"] = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    report["labels_missing_from_a_split"] = missing
    report["per_label_positives_total"] = {
        k: int(v) for k, v in df[labels].sum().sort_values(ascending=False).items()
    }

    # 8. Persist artifacts.
    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8")
    (OUT_DIR / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Console summary -----------------------------------------------------
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Raw rows ................ {report['raw_rows']:,}")
    print(f"Empty after normalize ... {report['removed_empty_after_norm']:,}")
    print(f"Exact duplicates dropped  {report['removed_exact_duplicates']:,}")
    print(f"Conflicting-text rows ... {report['conflicting_text_rows']:,}")
    print(f"Clean rows .............. {report['clean_rows']:,}")
    print(f"Labels .................. {report['num_labels']} (dropped {DEAD_LABELS})")
    print(
        f"Split ................... train {report['split_rows']['train']:,} | "
        f"val {report['split_rows']['val']:,} | test {report['split_rows']['test']:,}"
    )
    if missing:
        print(f"WARNING: labels missing from a split -> {missing}")
    else:
        print("Coverage ................ OK (all labels present in every split)")
    print(f"Artifacts ............... {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
