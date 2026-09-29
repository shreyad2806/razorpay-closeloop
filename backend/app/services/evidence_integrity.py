"""
Deterministic evidence hashing and integrity verification for Razorpay CloseLoop.

Provides stable SHA-256 content hashes over canonical JSON evidence payloads.
All operations are deterministic — no random salts, no timestamps in hash input.
"""

import hashlib
import json
from typing import Any, Dict


def compute_canonical_hash(payload: Dict[str, Any]) -> str:
    """
    Compute SHA-256 content hash of structured evidence payload.

    Canonicalization rules:
    - Excludes non-content transient metadata keys (`recorded_at`, `recorded_by`, `created_at`, `db_id`)
    - Sorts keys recursively
    - Compact separators (no trailing spaces)
    - UTF-8 encoding

    Args:
        payload: Dictionary containing evidence payload data

    Returns:
        64-character hexadecimal SHA-256 hash string
    """
    canonical_content = _canonicalize_dict(payload)
    canonical_json = json.dumps(
        canonical_content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _canonicalize_dict(obj: Any) -> Any:
    """Recursively strip transient keys and convert non-serializable values."""
    if isinstance(obj, dict):
        return {
            k: _canonicalize_dict(v)
            for k, v in sorted(obj.items())
            if k not in {"recorded_at", "recorded_by", "created_at", "db_id", "retrieved_at"}
        }
    elif isinstance(obj, list):
        return [_canonicalize_dict(item) for item in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
