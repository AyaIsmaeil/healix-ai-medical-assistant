"""
Healix - Parquet Writer (Phase 5, Stage 7).

Chunked, memory-bounded Parquet export. The full matrix is never materialised:
each chunk is converted to an Arrow table and appended through a single
``ParquetWriter`` handle.

Dtype policy (docs/research/DATASET_BUILDER_DESIGN.md §10.2):
  * booleans / one-hot -> ``int8`` **nullable** — 380 MB dense vs 3.04 GB as
    float64, and float64 would silently turn ``None`` into ``NaN``, destroying
    the null-vs-zero distinction the whole schema depends on.
  * ordinals -> ``int16`` nullable; continuous -> ``float32``.
Arrow keeps nulls as a real validity bitmap, so tri-state survives a round trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Columns that must stay wider than int8.
_FLOAT_COLUMNS = {"temperature_c", "positive_negative_ratio", "validity_score",
                  "interview_completeness_ratio"}
_INT16_COLUMNS = {"age", "onset_days_ago", "symptom_count"}


def arrow_schema(feature_order: List[str], extra: Dict[str, pa.DataType]) -> pa.Schema:
    fields: List[pa.Field] = []
    for name in feature_order:
        if name in _FLOAT_COLUMNS:
            dtype = pa.float32()
        elif name in _INT16_COLUMNS:
            dtype = pa.int16()
        else:
            dtype = pa.int8()
        fields.append(pa.field(name, dtype, nullable=True))
    for name, dtype in extra.items():
        fields.append(pa.field(name, dtype, nullable=True))
    return pa.schema(fields)


LABEL_FIELDS: Dict[str, pa.DataType] = {
    "y_disease": pa.string(),
    "y_disease_icd10": pa.string(),
    "y_urgency_prior": pa.string(),
    "y_specialty": pa.string(),
    "y_differential_ids": pa.list_(pa.string()),
    "y_differential_probs": pa.list_(pa.float32()),
    "differential_length": pa.int16(),
    "differential_top_id": pa.string(),
    "differential_top_prob": pa.float32(),
    "truth_in_differential": pa.bool_(),
    # QC annotations (never used as features)
    "content_hash": pa.string(),
    "duplicate_group_id": pa.string(),
    "leakage_flag": pa.int8(),
    "row_index": pa.int64(),
}


class ChunkedParquetWriter:
    """Append-only Parquet writer bounded by chunk size."""

    def __init__(self, path: Path, schema: pa.Schema, compression: str = "zstd"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._writer: Optional[pq.ParquetWriter] = None
        self._compression = compression
        self.rows_written = 0

    def write(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        columns = {
            field.name: pa.array([r.get(field.name) for r in rows], type=field.type)
            for field in self._schema
        }
        table = pa.Table.from_pydict(columns, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self._schema,
                                            compression=self._compression)
        self._writer.write_table(table)
        self.rows_written += len(rows)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ChunkedParquetWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def verify(path: Path, expected_rows: int, expected_columns: int) -> Dict[str, Any]:
    """Re-read the written file and assert its shape (export gate)."""
    meta = pq.read_metadata(path)
    ok = meta.num_rows == expected_rows and meta.num_columns == expected_columns
    return {
        "file": str(path),
        "rows": meta.num_rows,
        "columns": meta.num_columns,
        "size_bytes": Path(path).stat().st_size,
        "row_groups": meta.num_row_groups,
        "verified": bool(ok),
    }
