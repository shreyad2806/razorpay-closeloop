"""
Evaluation artifacts (Phase 15).

Writes small, deterministic JSON artifacts under ``backend/evaluation_results``.
No MLflow, no external experiment platform, no large binaries. Generated
artifacts are local outputs and are not meant to be committed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation_results"


def _default(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def write_artifacts(
    run_id: str,
    report: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Persist an evaluation report as a set of JSON artifacts.

    Returns a mapping of artifact name → absolute path written.
    """
    directory = Path(output_dir or DEFAULT_OUTPUT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    def _write(name: str, payload: Any) -> None:
        path = directory / name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_default)
        written[name] = str(path)

    _write(f"evaluation_run_{run_id}.json", report)
    _write("metrics.json", report.get("ml", {}))
    _write(
        "confusion_matrix.json",
        {
            "labels": report.get("ml", {}).get("confusion_labels", []),
            "matrix": report.get("ml", {}).get("confusion_matrix", []),
        },
    )
    _write("resolution_metrics.json", report.get("resolution", {}))
    _write("safety_metrics.json", report.get("safety", {}))
    _write("retrieval_metrics.json", report.get("retrieval", {}))
    _write("golden_results.json", report.get("golden", {}))
    _write("leakage_report.json", report.get("leakage", {}))
    _write("policy_report.json", report.get("policy", {}))
    _write("evidence_metrics.json", report.get("evidence", {}))

    return written


def new_run_id() -> str:
    """Deterministic-ish run identifier (timestamp based)."""
    return "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def artifacts_are_ignored() -> bool:
    """Whether the artifact directory is git-ignored (informational)."""
    return os.path.exists(DEFAULT_OUTPUT_DIR / ".gitignore")
