"""
Seed demo feedback records for the Learning page.

Creates realistic human review records that demonstrate the feedback loop:
Exception → Analysis → Recommendation → Human Decision → Feedback → Learning

These records populate the Learning & Feedback page with meaningful data.
"""

import sys
from pathlib import Path

# Ensure the backend root is on the path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.feedback import FeedbackService
from app.schemas.feedback import FeedbackType


def seed_demo_feedback():
    """Seed realistic demo feedback records via the FeedbackService singleton."""
    from app.api.dependencies import get_feedback_service
    svc = get_feedback_service()

    # Don't re-seed if already populated
    if sum(svc.count_by_type().values()) > 0:
        print(f"[SEED] Feedback already seeded: {svc.count_by_type()}")
        return svc

    # Demo feedback records — realistic human review decisions
    records = [
        # APPROVED — system correctly identified and resolved exceptions
        ("WF-DEMO-001", "CASE-DEMO-001", FeedbackType.APPROVE, "priya.rajesh@razorpay.com", "EXACT_MATCH"),
        ("WF-DEMO-002", "CASE-DEMO-002", FeedbackType.APPROVE, "priya.rajesh@razorpay.com", "EXACT_MATCH"),
        ("WF-DEMO-009", "CASE-DEMO-009", FeedbackType.APPROVE, "arun.kumar@razorpay.com", "TAX_ADJUSTMENT"),
        ("WF-DEMO-011", "CASE-DEMO-011", FeedbackType.APPROVE, "arun.kumar@razorpay.com", "FEE_DIFFERENCE"),
        ("WF-DEMO-025", "CASE-DEMO-025", FeedbackType.APPROVE, "priya.rajesh@razorpay.com", "EXACT_MATCH"),

        # REJECTED — system recommendation was incorrect
        ("WF-DEMO-013", "CASE-DEMO-013", FeedbackType.REJECT, "arun.kumar@razorpay.com", "DUPLICATE"),
        ("WF-DEMO-019", "CASE-DEMO-019", FeedbackType.REJECT, "priya.rajesh@razorpay.com", "UNKNOWN"),
        ("WF-DEMO-020", "CASE-DEMO-020", FeedbackType.REJECT, "arun.kumar@razorpay.com", "UNKNOWN"),

        # CORRECTED — system made an error that was fixed
        ("WF-DEMO-017", "CASE-DEMO-017", FeedbackType.CORRECT, "priya.rajesh@razorpay.com", "COMPLEX_MULTI_ADJUSTMENT"),

        # ESCALATED — sent to senior reviewer
        ("WF-DEMO-006", "CASE-DEMO-006", FeedbackType.ESCALATE, "arun.kumar@razorpay.com", "PARTIAL_SETTLEMENT"),
        ("WF-DEMO-015", "CASE-DEMO-015", FeedbackType.ESCALATE, "priya.rajesh@razorpay.com", "MISSING_RECORD"),
        ("WF-DEMO-028", "CASE-DEMO-028", FeedbackType.ESCALATE, "arun.kumar@razorpay.com", "DUPLICATE"),
        ("WF-DEMO-030", "CASE-DEMO-030", FeedbackType.ESCALATE, "priya.rajesh@razorpay.com", "UNKNOWN"),
    ]

    created = 0
    for wf_id, exc_id, fb_type, reviewer, prediction in records:
        svc.record_feedback(
            workflow_id=wf_id,
            exception_id=exc_id,
            feedback_type=fb_type,
            reviewer=reviewer,
            system_prediction=prediction,
        )
        created += 1
        print(f"  [{fb_type.value:8s}] {exc_id}")

    counts = svc.count_by_type()
    print(f"\n[SEED] Created {created} feedback records")
    print(f"  Approvals: {counts.get('APPROVE', 0)}")
    print(f"  Rejections: {counts.get('REJECT', 0)}")
    print(f"  Corrections: {counts.get('CORRECT', 0)}")
    print(f"  Escalations: {counts.get('ESCALATE', 0)}")

    return svc


if __name__ == "__main__":
    print("[SEED] Seeding demo feedback records...")
    seed_demo_feedback()
    print("[SEED] Done. Learning page will now show real metrics.")
