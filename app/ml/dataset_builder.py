"""
Healix - Dataset Builder (Phase 5) — OFFLINE ORCHESTRATOR.

Turns raw DDXPlus into ML-ready Healix datasets conforming to Feature Schema v2.

    DDXPlus CSV
        -> EvidenceParser        (token grammar -> nested raw values)
        -> OntologyMapper        (E_*/pathology -> HEALIX_* ids)
        -> FeatureEncoderV2      (schema-driven -> 286 columns)
        -> LabelBuilder          (4 targets)
        -> QualityValidator      (gates; flag-never-drop)
        -> StatisticsAccumulator (distributions, coverage, imbalance)
        -> ChunkedParquetWriter  (Parquet + manifests)

Run:
    python -m app.ml.dataset_builder --limit 5000      # smoke build
    python -m app.ml.dataset_builder                   # full build

This module is never imported by ``app.main``; it touches no runtime component.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from app.ml.dataset_manifest import (BUILDER_VERSION, build_manifest,
                                     compute_dataset_version, feature_dictionary,
                                     file_digest)
from app.ml.evidence_parser import EvidenceParseError, EvidenceParser
from app.ml.feature_encoder_v2 import FeatureEncoderV2
from app.ml.label_builder import LabelBuilder
from app.ml.ontology_mapper import OntologyMapper
from app.ml.parquet_writer import (LABEL_FIELDS, ChunkedParquetWriter,
                                   arrow_schema, verify)
from app.ml.quality_validator import QualityValidator, content_hash
from app.ml.statistics_report import StatisticsAccumulator

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DDXPLUS_DIR = REPO / "app" / "data" / "raw" / "ddxplus"
SCHEMA_PATH = REPO / "app" / "dictionaries" / "feature_schemas" / "v2.json"
OUTPUT_ROOT = REPO / "app" / "data" / "processed" / "ddxplus"

SPLIT_FILES = {"train": "train.csv", "validation": "validate.csv", "test": "test.csv"}
DEFAULT_CHUNK = 200_000
SEED = 20260721


class DatasetBuilder:
    """Deterministic, streaming, offline dataset builder."""

    def __init__(self, ddxplus_dir: Path = DDXPLUS_DIR,
                 schema_path: Path = SCHEMA_PATH,
                 output_root: Path = OUTPUT_ROOT,
                 chunk_size: int = DEFAULT_CHUNK,
                 limit: Optional[int] = None,
                 seed: int = SEED):
        self.ddxplus_dir = Path(ddxplus_dir)
        self.output_root = Path(output_root)
        self.chunk_size = chunk_size
        self.limit = limit
        self.seed = seed

        self.schema: Dict[str, Any] = json.loads(
            Path(schema_path).read_text(encoding="utf-8"))
        self.mapper = OntologyMapper.from_directory(self.ddxplus_dir)
        self.parser = EvidenceParser(self.mapper)
        self.encoder = FeatureEncoderV2(self.schema)
        self.labeller = LabelBuilder(self.mapper)

        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _iter_chunks(self, split: str):
        path = self.ddxplus_dir / SPLIT_FILES[split]
        seen = 0
        for chunk in pd.read_csv(path, chunksize=self.chunk_size, dtype=str,
                                 keep_default_na=False):
            if self.limit is not None:
                remaining = self.limit - seen
                if remaining <= 0:
                    return
                chunk = chunk.head(remaining)
            seen += len(chunk)
            yield chunk

    # ------------------------------------------------------------------
    def scan_hashes(self) -> Dict[str, Set[str]]:
        """Pass 1 — content hashes per split, for the cross-split leakage gate."""
        started = time.perf_counter()
        hashes: Dict[str, Set[str]] = {}
        for split in SPLIT_FILES:
            seen: Set[str] = set()
            for chunk in self._iter_chunks(split):
                for row in chunk.to_dict("records"):
                    seen.add(content_hash(row))
            hashes[split] = seen
            logger.info("hashed %s: %d unique records", split, len(seen))
        self.timings["leakage_scan"] = round(time.perf_counter() - started, 2)
        return hashes

    # ------------------------------------------------------------------
    def build_split(self, split: str, out_dir: Path,
                    leaked: Set[str]) -> Dict[str, Any]:
        """Pass 2 — parse, encode, label, validate, and export one split."""
        started = time.perf_counter()
        validator = QualityValidator(split, leaked_hashes=leaked)
        stats = StatisticsAccumulator(split, self.encoder.feature_order)
        schema = arrow_schema(self.encoder.feature_order, LABEL_FIELDS)
        quarantine: List[Dict[str, Any]] = []

        out_path = out_dir / "datasets" / f"{split}.parquet"
        row_index = 0

        with ChunkedParquetWriter(out_path, schema) as writer:
            for chunk in self._iter_chunks(split):
                buffer: List[Dict[str, Any]] = []
                for row in chunk.to_dict("records"):
                    try:
                        raw = self.parser.parse_row(row)
                        differential = self.parser.parse_differential(
                            row.get("DIFFERENTIAL_DIAGNOSIS"))
                        labels = self.labeller.build(row.get("PATHOLOGY", ""),
                                                     differential)
                        encoded = self.encoder.encode(raw)
                    except (EvidenceParseError, Exception) as exc:  # noqa: BLE001
                        validator.note_quarantine()
                        quarantine.append({"row_index": row_index,
                                           "reason": f"{type(exc).__name__}: {exc}"})
                        row_index += 1
                        continue

                    qc = validator.observe(row, encoded, labels)
                    stats.observe(encoded, labels, qc)

                    record = dict(encoded)
                    record.update(labels)
                    record.update(qc)
                    record["row_index"] = row_index
                    buffer.append(record)
                    row_index += 1
                writer.write(buffer)

            rows_written = writer.rows_written

        elapsed = round(time.perf_counter() - started, 2)
        self.timings[f"build_{split}"] = elapsed

        artifact = verify(out_path, rows_written,
                          len(self.encoder.feature_order) + len(LABEL_FIELDS))
        artifact["split"] = split

        if quarantine:
            qpath = out_dir / "quarantine" / f"{split}_rejected.json"
            qpath.parent.mkdir(parents=True, exist_ok=True)
            qpath.write_text(json.dumps(quarantine[:1000], indent=2), encoding="utf-8")

        return {"artifact": artifact,
                "quality": validator.report(),
                "statistics": stats.report(),
                "rows": rows_written,
                "seconds": elapsed}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        overall = time.perf_counter()

        # --- Gate: cross-split leakage must be measured BEFORE exporting ---
        hashes = self.scan_hashes()
        leaked = QualityValidator.compute_leaked_hashes(hashes)
        leakage_summary = {
            "definition": "blake2b-128 over AGE|SEX|EVIDENCES|INITIAL_EVIDENCE",
            "per_split_unique": {s: len(h) for s, h in hashes.items()},
            "leaked_unique_records": {s: len(v) for s, v in leaked.items()},
            "pairwise_shared_unique": {
                f"{a}__{b}": len(hashes[a] & hashes[b])
                for i, a in enumerate(hashes) for b in list(hashes)[i + 1:]
            },
            "policy": "flagged via leakage_flag; no rows removed; official split preserved",
        }

        digests = [file_digest(self.ddxplus_dir / f) for f in SPLIT_FILES.values()]
        digests += [file_digest(self.ddxplus_dir / n) for n in
                    ("release_evidences.json", "release_conditions.json")]
        dataset_version = compute_dataset_version(
            digests, self.schema.get("ontology_version", "unknown"),
            self.schema["schema_version"], self.seed)
        if self.limit:
            dataset_version += f"-smoke{self.limit}"

        out_dir = self.output_root / dataset_version
        for sub in ("datasets", "manifests", "metadata", "statistics"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        results = {s: self.build_split(s, out_dir, leaked[s]) for s in SPLIT_FILES}

        self.timings["total"] = round(time.perf_counter() - overall, 2)

        # ---------------- artifacts ----------------
        splits_meta = {s: {"rows": r["rows"], "source_file": SPLIT_FILES[s],
                           "quality": r["quality"]} for s, r in results.items()}
        manifest = build_manifest(
            dataset_version=dataset_version,
            ontology_version=self.schema.get("ontology_version", "unknown"),
            schema_version=self.schema["schema_version"],
            seed=self.seed, chunk_size=self.chunk_size,
            source_digests=digests, splits=splits_meta,
            artifacts=[r["artifact"] for r in results.values()],
            timings=self.timings)
        manifest["cross_split_leakage"] = leakage_summary

        metadata = {
            "dataset_version": dataset_version,
            "builder_version": BUILDER_VERSION,
            "ontology_version": self.schema.get("ontology_version"),
            "feature_schema_version": self.schema["schema_version"],
            "n_features": len(self.encoder.feature_order),
            "n_label_fields": len(LABEL_FIELDS),
            "splits": {s: r["rows"] for s, r in results.items()},
            "ontology_summary": self.mapper.summary(),
            "absence_semantics": "closed_world",
            "notes": [
                "Official DDXPlus split preserved; no reshuffling.",
                "temperature_c is null for every row (DDXPlus has only a fever boolean).",
                "sym_HEALIX_SYMPTOM_0097/0098 are null (no DDXPlus equivalent).",
                "y_specialty is derived from y_disease and adds no independent signal.",
                "y_urgency_prior is a DISEASE-LEVEL prior, not patient-level triage.",
                "No confidence target exists in DDXPlus.",
            ],
        }
        statistics = {s: r["statistics"] for s, r in results.items()}
        quality = {s: r["quality"] for s, r in results.items()}

        (out_dir / "manifests" / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        (out_dir / "metadata" / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        (out_dir / "statistics" / "statistics.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8")
        (out_dir / "statistics" / "quality_report.json").write_text(
            json.dumps(quality, indent=2), encoding="utf-8")
        (out_dir / "metadata" / "feature_dictionary.json").write_text(
            json.dumps(feature_dictionary(self.schema), indent=2), encoding="utf-8")
        (out_dir / "metadata" / "label_dictionary.json").write_text(
            json.dumps(LabelBuilder.label_dictionary(self.mapper), indent=2,
                       ensure_ascii=False), encoding="utf-8")
        (out_dir / "metadata" / "schema_snapshot.json").write_text(
            json.dumps(self.schema, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"dataset_version": dataset_version, "output_dir": str(out_dir),
                "manifest": manifest, "metadata": metadata,
                "statistics": statistics, "quality": quality}


def main() -> None:
    parser = argparse.ArgumentParser(description="Healix offline Dataset Builder")
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per split (smoke build)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    result = DatasetBuilder(chunk_size=args.chunk_size, limit=args.limit).run()
    print(json.dumps({
        "dataset_version": result["dataset_version"],
        "output_dir": result["output_dir"],
        "rows": result["metadata"]["splits"],
        "timings": result["manifest"]["timings_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
