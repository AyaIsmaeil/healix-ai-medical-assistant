#!/usr/bin/env python3
"""
Download DDXPlus (English) for Healix notebooks.

Source: HuggingFace mirror of the official Figshare release
  https://huggingface.co/datasets/aai530-group6/ddxplus
Paper: Tchango et al., NeurIPS Datasets & Benchmarks 2022 (arXiv:2205.09148)

Usage:
  python tools/download_ddxplus.py --quick    # validate.csv as train (~90 MB) — notebook dev
  python tools/download_ddxplus.py --full     # full train + validate + test (~850 MB)
  python tools/download_ddxplus.py --sample 50000   # first N rows only (after download)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "app" / "data" / "raw" / "ddxplus"
HF_REPO = "aai530-group6/ddxplus"

FILES_META = (
    "release_evidences.json",
    "release_conditions.json",
)


def _ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "  pip install huggingface_hub"
        ) from exc
    return hf_hub_download


def download_file(filename: str, dest: Path) -> Path:
    hf_hub_download = _ensure_hf_hub()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest.parent),
        )
    )
    if cached.resolve() != dest.resolve():
        shutil.copy2(cached, dest)
    print(f"  OK  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def maybe_unzip(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    extract_dir = path.parent
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)
    csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    path.unlink(missing_ok=True)
    if len(csv_members) == 1:
        extracted = extract_dir / csv_members[0]
        return extracted
    return path


def truncate_csv(src: Path, dest: Path, n_rows: int) -> None:
    import pandas as pd

    print(f"  truncating {src.name} -> {dest.name} ({n_rows:,} rows)...")
    chunk_iter = pd.read_csv(src, chunksize=min(n_rows, 50_000))
    first = next(chunk_iter)
    out = first.head(n_rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"  OK  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB, {len(out):,} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DDXPlus for Healix notebooks")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true",
                      help="download validate.csv (+ JSON) — enough to run notebooks locally")
    mode.add_argument("--full", action="store_true",
                      help="download train + validate + test + JSON")
    parser.add_argument("--sample", type=int, default=None,
                        help="after download, keep only first N rows in train.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {OUT_DIR}\n")

    for meta in FILES_META:
        dest = OUT_DIR / meta
        if not dest.exists():
            download_file(meta, dest)
        else:
            print(f"  skip {meta} (exists)")

    if args.quick:
        train_dest = OUT_DIR / "train.csv"
        if not train_dest.exists():
            # Smaller split — same schema as train; fine for notebook pipeline dev.
            validate_path = download_file("validate.csv", OUT_DIR / "validate.csv")
            shutil.copy2(validate_path, train_dest)
            print("  note: --quick uses validate.csv copied as train.csv for local notebook runs")
        else:
            print(f"  skip train.csv (exists)")
        if not (OUT_DIR / "validate.csv").exists():
            download_file("validate.csv", OUT_DIR / "validate.csv")

    if args.full:
        mapping = {
            "train.csv": OUT_DIR / "train.csv",
            "validate.csv": OUT_DIR / "validate.csv",
            "test.csv": OUT_DIR / "test.csv",
        }
        for remote, dest in mapping.items():
            if dest.exists():
                print(f"  skip {dest.name} (exists)")
                continue
            download_file(remote, dest)

    train_csv = OUT_DIR / "train.csv"
    if args.sample and train_csv.exists():
        tmp = OUT_DIR / "_train_full.csv"
        if not tmp.exists() and train_csv.stat().st_size > 0:
            shutil.copy2(train_csv, tmp)
        if tmp.exists():
            truncate_csv(tmp, train_csv, args.sample)

    if not train_csv.exists():
        print("ERROR: train.csv still missing.", file=sys.stderr)
        return 1

    print("\nDone. Next:")
    print("  1. Open training/notebooks/01_data_exploration.ipynb")
    print("  2. Run cells top to bottom through 07")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
