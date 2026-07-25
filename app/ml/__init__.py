"""
Healix - Offline ML tooling (Phase 5).

This package is **offline only**. It is never imported by ``app.main`` and never
participates in an HTTP request. It exists to turn raw research datasets
(DDXPlus first) into ML-ready artifacts that conform to Feature Schema v2.

Hard boundaries (see docs/research/DATASET_BUILDER_DESIGN.md §1):
  * It never imports ``app.domain.feature_encoder`` (the runtime encoder).
  * It never mutates the Assessment Engine, routes, or dictionaries.
  * It may use pandas / pyarrow — dependencies that are forbidden inside
    ``app/domain`` because that layer must stay pure.

Both the runtime encoder and this builder consume the *same* declarative
contract: ``app/dictionaries/feature_schemas/v2.json``.
"""
