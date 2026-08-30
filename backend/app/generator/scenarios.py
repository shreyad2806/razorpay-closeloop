"""
Centralized exception scenario definitions.

Each scenario defines:
- How financial records are modified
- Expected vs actual amounts
- Ground truth labels
- Resolvability
- Risk category

This module is the single source of truth for scenario behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.schemas.enums import (
    ExceptionType,
    MissingRecordSubtype,
    ResolutionType,
    RiskCategory,
)


@dataclass
class ScenarioDefinition:
    """Definition of an exception scenario."""

    exception_type: ExceptionType
    resolution: ResolutionType
    resolvable: bool
    risk_category: RiskCategory
    description: str

    # Which records this scenario modifies
    modifies_settlements: bool = False
    modifies_refunds: bool = False
    modifies_fees: bool = False
    modifies_taxes: bool = False
    modifies_adjustments: bool = False
    creates_duplicate: bool = False
    omits_record: bool = False
    adds_temporal_context: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Registry
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_DEFINITIONS: Dict[ExceptionType, ScenarioDefinition] = {
    # ── EXACT_MATCH ──────────────────────────────────────────────────────────
    ExceptionType.EXACT_MATCH: ScenarioDefinition(
        exception_type=ExceptionType.EXACT_MATCH,
        resolution=ResolutionType.NO_ACTION,
        resolvable=True,
        risk_category=RiskCategory.LOW,
        description="Expected equals actual. No discrepancy.",
    ),

    # ── FEE_DIFFERENCE ───────────────────────────────────────────────────────
    ExceptionType.FEE_DIFFERENCE: ScenarioDefinition(
        exception_type=ExceptionType.FEE_DIFFERENCE,
        resolution=ResolutionType.FEE_ADJUSTMENT,
        resolvable=True,
        risk_category=RiskCategory.MEDIUM,
        description=(
            "Settlement was computed with an incorrect fee amount. "
            "The fee record shows the correct amount, but settlement "
            "applied a different fee rate."
        ),
        modifies_settlements=True,
    ),

    # ── REFUND_ADJUSTMENT ────────────────────────────────────────────────────
    ExceptionType.REFUND_ADJUSTMENT: ScenarioDefinition(
        exception_type=ExceptionType.REFUND_ADJUSTMENT,
        resolution=ResolutionType.REFUND_ADJUSTMENT,
        resolvable=True,
        risk_category=RiskCategory.MEDIUM,
        description=(
            "Settlement does not account for a refund. "
            "The refund record exists but was not reflected in settlement."
        ),
        modifies_settlements=True,
    ),

    # ── TAX_ADJUSTMENT ───────────────────────────────────────────────────────
    ExceptionType.TAX_ADJUSTMENT: ScenarioDefinition(
        exception_type=ExceptionType.TAX_ADJUSTMENT,
        resolution=ResolutionType.TAX_ADJUSTMENT,
        resolvable=True,
        risk_category=RiskCategory.MEDIUM,
        description=(
            "Settlement was computed with an incorrect tax amount. "
            "The tax record shows the correct amount."
        ),
        modifies_settlements=True,
    ),

    # ── TIMING_DIFFERENCE ────────────────────────────────────────────────────
    ExceptionType.TIMING_DIFFERENCE: ScenarioDefinition(
        exception_type=ExceptionType.TIMING_DIFFERENCE,
        resolution=ResolutionType.TIMING_RECONCILIATION,
        resolvable=True,
        risk_category=RiskCategory.LOW,
        description=(
            "The settlement was observed before all financial records "
            "were available. The expected amount is correct but the "
            "actual settlement reflects an earlier point in time."
        ),
        adds_temporal_context=True,
    ),

    # ── PARTIAL_SETTLEMENT ───────────────────────────────────────────────────
    ExceptionType.PARTIAL_SETTLEMENT: ScenarioDefinition(
        exception_type=ExceptionType.PARTIAL_SETTLEMENT,
        resolution=ResolutionType.PARTIAL_SETTLEMENT_RECONCILIATION,
        resolvable=True,
        risk_category=RiskCategory.MEDIUM,
        description=(
            "Only a portion of the expected settlement was processed. "
            "The remaining amount is pending."
        ),
        modifies_settlements=True,
    ),

    # ── DUPLICATE ────────────────────────────────────────────────────────────
    ExceptionType.DUPLICATE: ScenarioDefinition(
        exception_type=ExceptionType.DUPLICATE,
        resolution=ResolutionType.DUPLICATE_SETTLEMENT,
        resolvable=True,
        risk_category=RiskCategory.HIGH,
        description=(
            "The same settlement was processed twice. "
            "Two settlement records exist for the same payment."
        ),
        creates_duplicate=True,
    ),

    # ── MISSING_RECORD ───────────────────────────────────────────────────────
    ExceptionType.MISSING_RECORD: ScenarioDefinition(
        exception_type=ExceptionType.MISSING_RECORD,
        resolution=ResolutionType.MISSING_RECORD_ESCALATION,
        resolvable=False,
        risk_category=RiskCategory.HIGH,
        description=(
            "A required financial record is missing. "
            "The expected amount cannot be verified."
        ),
        omits_record=True,
    ),

    # ── COMPLEX_MULTI_ADJUSTMENT ─────────────────────────────────────────────
    ExceptionType.COMPLEX_MULTI_ADJUSTMENT: ScenarioDefinition(
        exception_type=ExceptionType.COMPLEX_MULTI_ADJUSTMENT,
        resolution=ResolutionType.MULTI_ADJUSTMENT,
        resolvable=True,
        risk_category=RiskCategory.HIGH,
        description=(
            "Multiple legitimate events (refund + fee + tax adjustment) "
            "together explain the discrepancy."
        ),
        modifies_settlements=True,
        modifies_refunds=True,
        modifies_fees=True,
    ),

    # ── UNKNOWN ──────────────────────────────────────────────────────────────
    ExceptionType.UNKNOWN: ScenarioDefinition(
        exception_type=ExceptionType.UNKNOWN,
        resolution=ResolutionType.UNKNOWN_UNRESOLVED,
        resolvable=False,
        risk_category=RiskCategory.HIGH,
        description=(
            "An unexplained discrepancy that cannot be attributed "
            "to any known financial event."
        ),
    ),
}


def get_scenario_definition(exception_type: ExceptionType) -> ScenarioDefinition:
    """Get the scenario definition for an exception type."""
    return SCENARIO_DEFINITIONS[exception_type]


def get_all_scenario_types() -> List[ExceptionType]:
    """Get all defined scenario types."""
    return list(SCENARIO_DEFINITIONS.keys())
