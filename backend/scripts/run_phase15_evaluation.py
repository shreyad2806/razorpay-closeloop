"""
Run the Phase 15 evaluation harness and write its machine-readable artifacts.

Usage::

    cd backend
    python -m scripts.run_phase15_evaluation                 # full run, artifacts written
    python -m scripts.run_phase15_evaluation --per-family 10 --no-write

The script builds an isolated in-memory SQLite database so the database-backed
golden closed-loop scenarios can run without touching any real database, then
executes the complete evaluation twice to prove reproducibility.

Exit status is non-zero when the run is not reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Evaluation must never point at a real database.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///file:phase15_eval?mode=memory&cache=shared&uri=true"
)

import app.models  # noqa: E402,F401  (register every table before create_all)


def _session_factory():
    """A factory that yields a fresh isolated SQLite session per call."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from app.database.database import Base

    def make():
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )

        @event.listens_for(engine, "connect")
        def _pragma(dbapi_connection, _record):  # pragma: no cover - setup
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(bind=engine)
        return sessionmaker(bind=engine)()

    return make


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 15 evaluation.")
    parser.add_argument("--per-family", type=int, default=20, help="cases per family")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed")
    parser.add_argument("--no-write", action="store_true", help="skip artifact writing")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="artifact directory override"
    )
    args = parser.parse_args(argv)

    from app.evaluation.runner import reproducibility_check, run_evaluation

    factory = _session_factory()
    report = run_evaluation(
        per_family=args.per_family,
        seed=args.seed,
        session=factory,
        write=not args.no_write,
        output_dir=args.output_dir,
    )

    print("=== PHASE 15 EVALUATION HEADLINE ===")
    print(json.dumps(report.headline(), indent=2, sort_keys=True, default=str))
    print("\n=== SAFETY METRICS ===")
    print(json.dumps(report.safety, indent=2, sort_keys=True, default=str))
    print("\n=== GOLDEN CLOSED-LOOP SCENARIOS ===")
    for result in report.golden["results"]:
        expected = str(result["expected"]).encode("ascii", "replace").decode("ascii")
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {result['scenario_id']}: {status} - {expected}")
    print(f"  closure: {json.dumps(report.golden['closure'], sort_keys=True)}")

    comparison = reproducibility_check(per_family=max(3, args.per_family // 2), seed=args.seed)
    print("\n=== REPRODUCIBILITY ===")
    print(json.dumps({k: v for k, v in comparison.items() if k != "sections_compared"}, indent=2, default=str))
    return 0 if comparison["reproducible"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
