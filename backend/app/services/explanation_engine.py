"""
Deterministic evidence explanation engine for Razorpay CloseLoop.

Given an exception, determines whether available financial evidence
explains its discrepancy using controlled arithmetic and combination search.

All logic is deterministic. No ML, no LLM, no probabilistic reasoning.
"""

from itertools import combinations
from typing import Dict, List, Tuple

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import (
    CandidateExplanation,
    ExplainingEvent,
    ExplanationResult,
    ExplanationStatus,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of candidate events to consider for combination search
MAX_CANDIDATES = 20

# Maximum combination size to search (avoids exponential explosion)
MAX_COMBINATION_SIZE = 4

# Minimum absolute difference to consider (avoids floating-point-like noise in integer space)
MIN_DISCREPANCY = 1


class DeterministicExplanationEngine:
    """
    Deterministic explanation engine.

    Given an exception and its evidence package, determines whether
    available financial events explain the discrepancy.

    Algorithm:
    1. Calculate discrepancy = expected - actual
    2. Collect evidence events with their contributions
    3. Search for subset sums that equal the discrepancy
    4. Classify as FULLY_EXPLAINED, PARTIALLY_EXPLAINED, UNEXPLAINED, or CONFLICTING
    """

    def explain(
        self, package: EvidencePackage, graph=None
    ) -> ExplanationResult:
        """
        Produce a deterministic explanation for an exception.

        Args:
            package: The EvidencePackage from evidence retrieval
            graph: Optional NetworkX graph (not required, but used for validation)

        Returns:
            ExplanationResult with explanation status and evidence
        """
        difference = package.difference

        # 1. Collect candidate events with contributions
        candidates = self._collect_candidates(package)

        # 2. Check for zero discrepancy (exact match)
        if abs(difference) < MIN_DISCREPANCY:
            return self._build_exact_match(package, candidates)

        # 3. Search for subset sum explanations
        exact_matches = self._find_exact_explanations(difference, candidates)

        # 4. Find best partial explanation
        partial_amount, partial_ids = self._find_partial_explanation(difference, candidates)

        # 5. Check for conflicts
        has_conflict = len(exact_matches) > 1

        # 6. Determine status and build result
        if has_conflict:
            return self._build_conflicting(package, exact_matches, candidates)
        elif len(exact_matches) == 1:
            return self._build_fully_explained(package, exact_matches[0], candidates)
        elif partial_amount != 0:
            return self._build_partially_explained(
                package, partial_amount, partial_ids, candidates
            )
        else:
            return self._build_unexplained(package, candidates)

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate Collection
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_candidates(self, package: EvidencePackage) -> List[Dict]:
        """
        Collect evidence events as candidates for explanation.

        Returns list of dicts with record_id, entity_type, amount, contribution.
        """
        candidates = []

        # Refunds: contribution = -amount (reduce expected)
        for r in package.refunds:
            candidates.append({
                "record_id": r.record_id,
                "entity_type": "REFUND",
                "amount": r.amount,
                "contribution": -r.amount,
            })

        # Fees: contribution = -amount (reduce expected)
        for f in package.fees:
            candidates.append({
                "record_id": f.record_id,
                "entity_type": "FEE",
                "amount": f.amount,
                "contribution": -f.amount,
            })

        # Taxes: contribution = -amount (reduce expected)
        for t in package.taxes:
            candidates.append({
                "record_id": t.record_id,
                "entity_type": "TAX",
                "amount": t.amount,
                "contribution": -t.amount,
            })

        # Adjustments: contribution = amount (signed)
        for a in package.adjustments:
            candidates.append({
                "record_id": a.record_id,
                "entity_type": "ADJUSTMENT",
                "amount": a.amount,
                "contribution": a.amount,
            })

        # Limit candidates to prevent exponential explosion
        if len(candidates) > MAX_CANDIDATES:
            # Keep largest contributions first (most likely to explain)
            candidates.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            candidates = candidates[:MAX_CANDIDATES]

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Exact Match (zero discrepancy)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_exact_match(
        self, package: EvidencePackage, candidates: List[Dict]
    ) -> ExplanationResult:
        """Build result for zero-discrepancy case."""
        return ExplanationResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            explanation_status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=0,
            remaining_difference=0,
            supporting_evidence_ids=[],
            candidate_explanations=[],
            conflict=False,
            missing_evidence=[m.entity_type for m in package.missing_evidence],
            explanation_reason="No discrepancy detected. Expected and actual amounts match.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Subset Sum Search
    # ─────────────────────────────────────────────────────────────────────────

    def _find_exact_explanations(
        self, difference: int, candidates: List[Dict]
    ) -> List[List[Dict]]:
        """
        Find all combinations of candidates whose total contribution equals the discrepancy.

        Uses controlled combination search up to MAX_COMBINATION_SIZE.
        Returns list of matching combinations (each is a list of candidate dicts).
        """
        if not candidates:
            return []

        exact_matches = []

        # Search combinations of size 1 to MAX_COMBINATION_SIZE
        max_size = min(len(candidates), MAX_COMBINATION_SIZE)

        for size in range(1, max_size + 1):
            for combo in combinations(candidates, size):
                total = sum(c["contribution"] for c in combo)
                if total == difference:
                    exact_matches.append(list(combo))

        return exact_matches

    def _find_partial_explanation(
        self, difference: int, candidates: List[Dict]
    ) -> Tuple[int, List[str]]:
        """
        Find the best partial explanation — the combination that gets closest
        to explaining the discrepancy without overshooting.

        Returns (explained_amount, list_of_record_ids).
        """
        if not candidates:
            return 0, []

        best_amount = 0
        best_ids = []

        # Try single events first
        for c in candidates:
            if abs(c["contribution"]) <= abs(difference):
                # This event's contribution is within the discrepancy
                if abs(c["contribution"]) > abs(best_amount):
                    best_amount = c["contribution"]
                    best_ids = [c["record_id"]]

        # Try pairs if single event doesn't fully explain
        if abs(best_amount) < abs(difference) and len(candidates) >= 2:
            max_size = min(len(candidates), MAX_COMBINATION_SIZE)
            for size in range(2, max_size + 1):
                for combo in combinations(candidates, size):
                    total = sum(c["contribution"] for c in combo)
                    # Check if this combo is closer to difference without overshooting
                    if (
                        abs(total) <= abs(difference)
                        and abs(total) > abs(best_amount)
                    ):
                        best_amount = total
                        best_ids = [c["record_id"] for c in combo]

        return best_amount, best_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Result Builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fully_explained(
        self,
        package: EvidencePackage,
        explanation_events: List[Dict],
        all_candidates: List[Dict],
    ) -> ExplanationResult:
        """Build result for fully explained case."""
        event_ids = [e["record_id"] for e in explanation_events]
        reason = self._generate_reason(explanation_events, package.difference)

        candidate = CandidateExplanation(
            events=[
                ExplainingEvent(
                    record_id=e["record_id"],
                    entity_type=e["entity_type"],
                    amount=e["amount"],
                    contribution=e["contribution"],
                )
                for e in explanation_events
            ],
            total_contribution=sum(e["contribution"] for e in explanation_events),
            is_exact_match=True,
        )

        return ExplanationResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            explanation_status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=package.difference,
            remaining_difference=0,
            supporting_evidence_ids=event_ids,
            candidate_explanations=[candidate],
            conflict=False,
            missing_evidence=[m.entity_type for m in package.missing_evidence],
            explanation_reason=reason,
        )

    def _build_partially_explained(
        self,
        package: EvidencePackage,
        explained_amount: int,
        explained_ids: List[str],
        all_candidates: List[Dict],
    ) -> ExplanationResult:
        """Build result for partially explained case."""
        remaining = package.difference - explained_amount

        explained_events = [
            c for c in all_candidates if c["record_id"] in explained_ids
        ]
        reason = self._generate_partial_reason(
            explained_events, explained_amount, remaining
        )

        candidate = CandidateExplanation(
            events=[
                ExplainingEvent(
                    record_id=e["record_id"],
                    entity_type=e["entity_type"],
                    amount=e["amount"],
                    contribution=e["contribution"],
                )
                for e in explained_events
            ],
            total_contribution=explained_amount,
            is_exact_match=False,
        )

        return ExplanationResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            explanation_status=ExplanationStatus.PARTIALLY_EXPLAINED,
            explained_amount=explained_amount,
            remaining_difference=remaining,
            supporting_evidence_ids=explained_ids,
            candidate_explanations=[candidate],
            conflict=False,
            missing_evidence=[m.entity_type for m in package.missing_evidence],
            explanation_reason=reason,
        )

    def _build_unexplained(
        self, package: EvidencePackage, all_candidates: List[Dict]
    ) -> ExplanationResult:
        """Build result for unexplained case."""
        missing_types = [m.entity_type for m in package.missing_evidence]

        if missing_types:
            reason = (
                f"Cannot explain ₹{abs(package.difference) // 100} discrepancy. "
                f"Missing evidence: {', '.join(missing_types)}."
            )
        else:
            reason = (
                f"Cannot explain ₹{abs(package.difference) // 100} discrepancy. "
                f"No available evidence accounts for the difference."
            )

        return ExplanationResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            explanation_status=ExplanationStatus.UNEXPLAINED,
            explained_amount=0,
            remaining_difference=package.difference,
            supporting_evidence_ids=[],
            candidate_explanations=[],
            conflict=False,
            missing_evidence=missing_types,
            explanation_reason=reason,
        )

    def _build_conflicting(
        self,
        package: EvidencePackage,
        exact_matches: List[List[Dict]],
        all_candidates: List[Dict],
    ) -> ExplanationResult:
        """Build result for conflicting explanations."""
        candidates = []
        all_ids = set()

        for match in exact_matches:
            event_ids = [e["record_id"] for e in match]
            all_ids.update(event_ids)
            candidates.append(
                CandidateExplanation(
                    events=[
                        ExplainingEvent(
                            record_id=e["record_id"],
                            entity_type=e["entity_type"],
                            amount=e["amount"],
                            contribution=e["contribution"],
                        )
                        for e in match
                    ],
                    total_contribution=sum(e["contribution"] for e in match),
                    is_exact_match=True,
                )
            )

        reason = (
            f"Multiple explanations found for ₹{abs(package.difference) // 100} discrepancy. "
            f"{len(candidates)} candidate combinations detected. "
            f"Deterministic preference cannot be established."
        )

        return ExplanationResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            explanation_status=ExplanationStatus.CONFLICTING,
            explained_amount=package.difference,
            remaining_difference=0,
            supporting_evidence_ids=list(all_ids),
            candidate_explanations=candidates,
            conflict=True,
            missing_evidence=[m.entity_type for m in package.missing_evidence],
            explanation_reason=reason,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Reason Generation (template-based, no LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_reason(
        self, events: List[Dict], difference: int
    ) -> str:
        """Generate a deterministic template-based explanation reason."""
        if not events:
            return "No evidence events found."

        parts = []
        for e in events:
            entity = e["entity_type"].lower()
            amount_str = f"₹{abs(e['amount']) // 100}"
            parts.append(f"{entity} {e['record_id']} for {amount_str}")

        if len(parts) == 1:
            explanation = parts[0]
        elif len(parts) == 2:
            explanation = f"{parts[0]} and {parts[1]}"
        else:
            explanation = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        diff_str = f"₹{abs(difference) // 100}"
        return f"{explanation.capitalize()} explain the {diff_str} discrepancy."

    def _generate_partial_reason(
        self, events: List[Dict], explained: int, remaining: int
    ) -> str:
        """Generate a deterministic template-based partial explanation reason."""
        if not events:
            return "No evidence events found to explain the discrepancy."

        parts = []
        for e in events:
            entity = e["entity_type"].lower()
            amount_str = f"₹{abs(e['amount']) // 100}"
            parts.append(f"{entity} {e['record_id']} for {amount_str}")

        if len(parts) == 1:
            explanation = parts[0]
        elif len(parts) == 2:
            explanation = f"{parts[0]} and {parts[1]}"
        else:
            explanation = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        explained_str = f"₹{abs(explained) // 100}"
        remaining_str = f"₹{abs(remaining) // 100}"
        return (
            f"{explanation.capitalize()} partially explain the discrepancy. "
            f"Explained: {explained_str}. Remaining: {remaining_str}."
        )
