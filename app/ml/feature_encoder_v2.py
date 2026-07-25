"""
Healix - Feature Encoder v2 (Phase 5, Stage 3) — OFFLINE.

Encodes the nested raw-value mapping produced by ``EvidenceParser`` into the
286 named columns declared by ``app/dictionaries/feature_schemas/v2.json``.

This is deliberately **schema-driven**: every column is resolved by following
the ``source`` dotted path declared in the schema. The runtime
``app.domain.feature_encoder.FeatureEncoder`` currently hardcodes its six raw
values and ignores those ``source`` strings (blocker **B1**); this module is
therefore also the *reference implementation* of the resolver that B1 will need.

Encoding semantics are identical to the runtime encoder (verified against
``_encode_boolean`` / ``_encode_categorical``):
  * boolean  -> 1 / 0 / None            (tri-state; unknown stays unknown)
  * numeric  -> value unchanged / None  (no imputation, ever)
  * categorical -> one-hot; **unknown or unmapped value => every member None**
    (never zeroed, which would fabricate "confirmed not-X")

This module never imports the runtime encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SchemaError(RuntimeError):
    """Raised when the schema and the produced columns disagree."""


def resolve_path(raw: Dict[str, Any], path: str) -> Any:
    """Follow a dotted ``source`` path; missing segments yield ``None``."""
    node: Any = raw
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


class FeatureEncoderV2:
    """Schema-driven encoder for the offline dataset builder."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema_version: str = schema["schema_version"]
        self.ontology_version: Optional[str] = schema.get("ontology_version")
        self.feature_order: List[str] = list(schema["feature_order"])
        self._numeric: Dict[str, Any] = schema.get("numeric_fields", {})
        self._boolean: Dict[str, Any] = schema.get("boolean_fields", {})
        self._categorical: Dict[str, Any] = schema.get("categorical_fields", {})

        produced = list(self._numeric) + list(self._boolean)
        for spec in self._categorical.values():
            produced += [f"{spec['one_hot_prefix']}_{v}" for v in spec["values"]]
        if set(produced) != set(self.feature_order):
            missing = sorted(set(produced) - set(self.feature_order))
            extra = sorted(set(self.feature_order) - set(produced))
            raise SchemaError(
                f"feature_order mismatch — missing: {missing} | extra: {extra}")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise SchemaError("feature_order contains duplicates")

    # ------------------------------------------------------------------
    def encode(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Return one fully-populated row: every declared column is present."""
        columns: Dict[str, Any] = {}

        for name, spec in self._numeric.items():
            columns[name] = self._numeric_value(resolve_path(raw, spec["source"]))

        for name, spec in self._boolean.items():
            columns[name] = self._boolean_value(resolve_path(raw, spec["source"]))

        for spec in self._categorical.values():
            columns.update(self._one_hot(
                resolve_path(raw, spec["source"]), spec["values"],
                spec["one_hot_prefix"]))

        # Deterministic order — identical input always yields identical key order.
        return {name: columns[name] for name in self.feature_order}

    # ------------------------------------------------------------------
    @staticmethod
    def _numeric_value(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        return value if isinstance(value, (int, float)) else None

    @staticmethod
    def _boolean_value(value: Any) -> Optional[int]:
        if value is None:
            return None
        return 1 if value else 0

    @staticmethod
    def _one_hot(value: Any, allowed: List[str],
                 prefix: str) -> Dict[str, Optional[int]]:
        columns: Dict[str, Optional[int]] = {f"{prefix}_{v}": None for v in allowed}
        if value is None or value not in allowed:
            return columns          # unknown -> all null (never 0)
        for candidate in allowed:
            columns[f"{prefix}_{candidate}"] = 1 if candidate == value else 0
        return columns
