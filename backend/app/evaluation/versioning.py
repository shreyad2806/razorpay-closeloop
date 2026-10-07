"""
Version pins for evaluation (Phase 15).

Every evaluation run records these so results are attributable and
reproducible. Bumping a dataset or evaluator version is how a previous run is
invalidated.
"""

from __future__ import annotations

#: Version of the evaluation harness itself. Bump on any metric change.
EVALUATOR_VERSION = "15.0.0"

#: Synthetic evaluation dataset versions.
SYNTHETIC_DATASET_V1 = "eval-synthetic-v1"

#: Tag used for real historical evaluation datasets.
REAL_DATASET_TAG = "eval-real-v1"

#: Dataset provenance markers. Synthetic and real data are never mixed silently.
DATASET_SOURCE_SYNTHETIC = "SYNTHETIC"
DATASET_SOURCE_REAL = "REAL_HISTORICAL"

#: Default deterministic seed for all evaluation randomness.
DEFAULT_EVAL_SEED = 42

#: Synthetic dataset schema version (independent of feature schema).
SYNTHETIC_DATASET_SCHEMA_VERSION = "1.0.0"


def dataset_fingerprint(payload: str) -> str:
    """Stable content fingerprint for a canonicalized dataset payload."""
    import hashlib

    return "ds-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
