# Razorpay CloseLoop — Financial Data Contract

## Overview

This document defines the synthetic financial data contract for the Razorpay CloseLoop reconciliation system. All schemas live in `backend/app/schemas/`.

## Entities

| Entity | ID Prefix | Description |
|--------|-----------|-------------|
| Merchant | `MER-` | Payment gateway merchant |
| Payment | `PAY-` | Captured payment transaction |
| Settlement | `SET-` | Payout to merchant |
| Refund | `REF-` | Refund against a payment |
| Fee | `FEE-` | Charge applied to a payment |
| Tax | `TAX-` | Tax on payment or fee |
| Adjustment | `ADJ-` | Financial correction |
| Case | `CASE-` | Reconciliation situation |
| Ground Truth | — | Answer key (by case_id) |

## Relationships

```
Merchant (1) ──< (N) Payment
Payment  (1) ──< (N) Settlement
Payment  (1) ──< (N) Refund
Payment  (1) ──< (N) Fee
Payment  (1) ──< (N) Tax
Payment  (1) ──< (N) Adjustment
Case     (1) ─── (1) Ground Truth
Case     (1) ─── (1) Payment
```

Every Settlement, Refund, Fee, Tax, and Adjustment links to a Payment via `payment_id` and optionally to a Case via `case_id`.

## Amount Representation

All monetary amounts use **integer minor units (paise)**. No floating-point values are used in financial calculations.

Example: ₹150.75 = `15075` paise

## Expected Settlement Formula

```
expected_settlement = payment_amount
                    - total_refunds
                    - total_fees
                    - total_taxes
                    + total_adjustments

difference = actual_settlement - expected_settlement
```

## Exception Taxonomy (`ExceptionType`)

| Type | Description |
|------|-------------|
| `EXACT_MATCH` | Expected equals actual, no action needed |
| `FEE_DIFFERENCE` | Discrepancy caused by fee miscalculation |
| `REFUND_ADJUSTMENT` | Discrepancy caused by refund timing or amount |
| `TAX_ADJUSTMENT` | Discrepancy caused by tax calculation |
| `TIMING_DIFFERENCE` | Same transaction, different observation windows |
| `PARTIAL_SETTLEMENT` | Settlement covers only part of the payment |
| `DUPLICATE` | Multiple settlements for the same payment |
| `MISSING_RECORD` | Expected record absent from settlement data |
| `COMPLEX_MULTI_ADJUSTMENT` | Multiple factors causing discrepancy |
| `UNKNOWN` | Unrecognized or ambiguous pattern |

## Resolution Taxonomy (`ResolutionType`)

| Type | Description |
|------|-------------|
| `NO_ACTION` | Exact match, nothing to do |
| `FEE_ADJUSTMENT` | Correct fee amount |
| `REFUND_ADJUSTMENT` | Correct refund amount |
| `TAX_ADJUSTMENT` | Correct tax amount |
| `TIMING_RECONCILIATION` | Align timing across systems |
| `PARTIAL_SETTLEMENT_RECONCILIATION` | Complete partial settlement |
| `DUPLICATE_SETTLEMENT` | Reverse duplicate settlement |
| `MISSING_RECORD_ESCALATION` | Escalate missing record |
| `MULTI_ADJUSTMENT` | Apply multiple corrections |
| `UNKNOWN_UNRESOLVED` | Cannot determine resolution |

## Risk Categories (`RiskCategory`)

| Level | Description |
|-------|-------------|
| `LOW` | Known, small, easily resolvable |
| `MEDIUM` | Moderate impact or complexity |
| `HIGH` | Large discrepancy, unknown pattern, or high impact |

Risk is independent of exception type.

## Ground Truth

Ground truth is stored **separately** from generated financial records and model outputs.

Location: `data/ground_truth/`

Each ground truth record contains:
- Financial breakdown (payment, refunds, fees, taxes, adjustments)
- Expected vs actual amounts
- True exception type and resolution
- Resolvability and risk category

The `verify_expected_amount()` method validates the calculation:
`payment - refunds - fees - taxes + adjustments == expected`

## Dataset Storage

```
data/
├── raw/              # Source financial events (if imported)
├── generated/        # Generated synthetic financial records
│   ├── merchants.json
│   ├── payments.json
│   ├── settlements.json
│   ├── refunds.json
│   ├── fees.json
│   ├── taxes.json
│   ├── adjustments.json
│   └── cases.json
└── ground_truth/     # Definitive answer key
    └── ground_truth.json
```

## Deterministic Generation

The `GeneratorConfig` contract controls reproducibility:

- `random_seed`: Fixed seed for identical output
- `num_merchants`: Number of merchants
- `num_cases`: Number of reconciliation cases
- `scenario_distribution`: Weighted distribution of exception types
- `date_range_start` / `date_range_end`: Event date bounds
- `currency`: Currency for all amounts
- `min/max_payment_amount_paise`: Payment amount bounds
- `fee_rate_bps` / `tax_rate_bps`: Fee and tax rates in basis points
- `duplicate_probability` / `missing_record_probability`: Anomaly rates

Same seed + same config = same dataset, always.

## Source Files

| File | Contents |
|------|----------|
| `schemas/enums.py` | All enums: ExceptionType, ResolutionType, RiskCategory, statuses, types |
| `schemas/financial.py` | Entity models: Merchant, Payment, Settlement, Refund, Fee, Tax, Adjustment |
| `schemas/case.py` | Case and GroundTruth models |
| `schemas/config.py` | GeneratorConfig and ScenarioDistribution |
| `schemas/__init__.py` | Public exports |
