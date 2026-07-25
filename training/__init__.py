"""
Healix - offline training packages (Phase 6).

Everything under ``training/`` is completely isolated from the runtime
application (``app/``): it is never imported by ``app.main``, never touches
FastAPI routes, and never replaces the Rule-Based Assessment Engine adapters.
It exists purely to build, evaluate, and compare candidate ML models against
the Parquet datasets produced by the Phase 5 Dataset Builder.
"""
