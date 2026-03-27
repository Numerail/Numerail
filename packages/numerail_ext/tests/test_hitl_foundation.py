"""Tests for Stage 1 HITL foundation: types, triggers, and local gateway.

Covers:
  - HumanDecisionAction enum values
  - HumanDecision frozen dataclass (immutability, modified_vector, guidance)
  - ReviewOutcome enum values
  - HumanReviewTriggers profile presets (ADVISORY, SUPERVISORY, MANDATORY)
  - HumanReviewTriggers custom construction
  - evaluate_triggers(): no-fire, single-trigger, multi-trigger scenarios
  - highest_priority_trigger(): BLOCKING over NOTIFY, audit over reject, empty
  - LocalApprovalGateway: submit, poll, cancel, program_decision, auto modes
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from numerail.protocols import (
    ApprovalGateway,
    HumanDecision,
    HumanDecisionAction,
    ReviewOutcome,
)
from numerail_ext.survivability.hitl import (
    HumanReviewProfile,
    HumanReviewTriggers,
    ReviewMode,
)
from numerail_ext.survivability.local_gateway import LocalApprovalGateway


# ── Helpers ───────────────────────────────────────────────────────────────


def _now_ms() -> float:
    return float(time.time_ns() // 1_000_000)


def _decision(
    review_id: str = "review-0",
    action: HumanDecisionAction = HumanDecisionAction.APPROVE,
    reviewer_id: str = "tester",
    authenticated: bool = True,
    reason: str = "test decision",
    **kwargs,
) -> HumanDecision:
    return HumanDecision(
        review_id=review_id,
        action=action,
        reviewer_id=reviewer_id,
        authenticated=authenticated,
        timestamp_ms=_now_ms(),
        reason=reason,
        **kwargs,
    )


class _FakeOutput:
    """Minimal stand-in for EnforcementOutput in gateway tests."""

    def __init__(self, result: str = "reject"):
        self.result = result

    def __str__(self) -> str:
        return self.result


def _base_triggers() -> dict:
    """Default all-clean inputs for evaluate_triggers."""
    return dict(
        enforcement_result=None,
        routing_decision=None,
        breaker_mode_changed=False,
        safe_stop_entered=False,
        budget_exhausted=False,
        budget_remaining_fraction=None,
        policy_version_changed=False,
        audit_chain_failed=False,
        trusted_field_mismatch_fraction=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TYPE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanDecisionAction:
    def test_human_decision_action_values(self):
        """All five enum values exist and have the correct string values."""
        assert HumanDecisionAction.APPROVE == "approve"
        assert HumanDecisionAction.DENY == "deny"
        assert HumanDecisionAction.MODIFY == "modify"
        assert HumanDecisionAction.ESCALATE == "escalate"
        assert HumanDecisionAction.DEFER == "defer"
        assert len(HumanDecisionAction) == 5


class TestHumanDecision:
    def test_human_decision_frozen(self):
        """HumanDecision is immutable — assigning to any field raises."""
        d = _decision()
        with pytest.raises((AttributeError, TypeError)):
            d.reason = "mutated"

    def test_human_decision_with_modified_vector(self):
        """MODIFY decision with numpy array round-trips correctly."""
        vec = np.array([1.0, 2.0, 3.0])
        d = _decision(action=HumanDecisionAction.MODIFY, modified_vector=vec)
        assert d.action == HumanDecisionAction.MODIFY
        assert d.modified_vector is not None
        np.testing.assert_array_equal(d.modified_vector, vec)

    def test_human_decision_with_guidance(self):
        """DENY decision with guidance dict is accessible."""
        guidance = {"amount": (0.0, 100.0), "rate": (0.01, 0.05)}
        d = _decision(action=HumanDecisionAction.DENY, guidance=guidance)
        assert d.action == HumanDecisionAction.DENY
        assert d.guidance == {"amount": (0.0, 100.0), "rate": (0.01, 0.05)}

    def test_human_decision_default_fields(self):
        """Optional fields default to None / 0 correctly."""
        d = _decision()
        assert d.modified_vector is None
        assert d.guidance is None
        assert d.escalation_depth == 0

    def test_human_decision_authenticated_false(self):
        """authenticated=False is representable."""
        d = _decision(authenticated=False)
        assert d.authenticated is False


class TestReviewOutcome:
    def test_review_outcome_values(self):
        """All four enum values exist and have correct string values."""
        assert ReviewOutcome.EXECUTED == "executed"
        assert ReviewOutcome.PENDING_REVIEW == "pending_review"
        assert ReviewOutcome.DENIED == "denied"
        assert ReviewOutcome.EXPIRED == "expired"
        assert len(ReviewOutcome) == 4


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestAdvisoryProfile:
    def test_advisory_profile(self):
        """ADVISORY: safe_stop_entry is BLOCKING, reject is None, timeout 600."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.ADVISORY)
        assert t.on_safe_stop_entry == ReviewMode.BLOCKING
        assert t.on_audit_chain_failure == ReviewMode.BLOCKING
        assert t.on_budget_exhaustion == ReviewMode.NOTIFY
        assert t.on_reject is None
        assert t.review_timeout_seconds == 600.0
        assert t.max_escalation_depth == 1

    def test_advisory_nones(self):
        """ADVISORY: non-configured triggers are all None."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.ADVISORY)
        assert t.on_reject is None
        assert t.on_project_flagged is None
        assert t.on_project_confirmation_required is None
        assert t.on_breaker_mode_change is None
        assert t.on_policy_version_change is None


class TestSupervisoryProfile:
    def test_supervisory_profile(self):
        """SUPERVISORY: reject is BLOCKING, breaker_mode_change is NOTIFY, timeout 300."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        assert t.on_reject == ReviewMode.BLOCKING
        assert t.on_project_confirmation_required == ReviewMode.BLOCKING
        assert t.on_projection_forbidden_reject == ReviewMode.BLOCKING
        assert t.on_safe_stop_entry == ReviewMode.BLOCKING
        assert t.on_budget_exhaustion == ReviewMode.BLOCKING
        assert t.on_audit_chain_failure == ReviewMode.BLOCKING
        assert t.on_breaker_mode_change == ReviewMode.NOTIFY
        assert t.on_project_flagged == ReviewMode.NOTIFY
        assert t.on_budget_low_remaining_fraction == pytest.approx(0.10)
        assert t.on_budget_low_mode == ReviewMode.NOTIFY
        assert t.review_timeout_seconds == 300.0
        assert t.max_escalation_depth == 2


class TestMandatoryProfile:
    def test_mandatory_profile(self):
        """MANDATORY: project_flagged is BLOCKING, timeout 120."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.MANDATORY)
        assert t.on_reject == ReviewMode.BLOCKING
        assert t.on_project_confirmation_required == ReviewMode.BLOCKING
        assert t.on_project_flagged == ReviewMode.BLOCKING
        assert t.on_projection_forbidden_reject == ReviewMode.BLOCKING
        assert t.on_breaker_mode_change == ReviewMode.BLOCKING
        assert t.on_safe_stop_entry == ReviewMode.BLOCKING
        assert t.on_budget_exhaustion == ReviewMode.BLOCKING
        assert t.on_policy_version_change == ReviewMode.BLOCKING
        assert t.on_audit_chain_failure == ReviewMode.BLOCKING
        assert t.on_budget_low_remaining_fraction == pytest.approx(0.20)
        assert t.on_budget_low_mode == ReviewMode.NOTIFY
        assert t.review_timeout_seconds == 120.0
        assert t.max_escalation_depth == 2


class TestCustomTriggers:
    def test_custom_triggers(self):
        """Custom construction overrides all defaults independently."""
        t = HumanReviewTriggers(
            on_reject=ReviewMode.NOTIFY,
            review_timeout_seconds=60.0,
            max_escalation_depth=3,
        )
        assert t.on_reject == ReviewMode.NOTIFY
        assert t.on_safe_stop_entry is None
        assert t.review_timeout_seconds == 60.0
        assert t.max_escalation_depth == 3


# ═══════════════════════════════════════════════════════════════════════════
#  TRIGGER EVALUATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateTriggers:
    def test_evaluate_no_triggers_fire(self):
        """All inputs clean → empty fired list."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        fired = t.evaluate_triggers(**_base_triggers())
        assert fired == []

    def test_evaluate_reject_fires_supervisory(self):
        """REJECT under SUPERVISORY → (on_reject, BLOCKING) in fired list."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "enforcement_result": "reject"}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_reject", ReviewMode.BLOCKING) in fired

    def test_evaluate_reject_does_not_fire_advisory(self):
        """REJECT under ADVISORY → on_reject not in fired list (it's None)."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.ADVISORY)
        inputs = {**_base_triggers(), "enforcement_result": "reject"}
        fired = t.evaluate_triggers(**inputs)
        trigger_names = [name for name, _ in fired]
        assert "on_reject" not in trigger_names

    def test_evaluate_safe_stop_fires_advisory(self):
        """safe_stop_entered=True fires under ADVISORY."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.ADVISORY)
        inputs = {**_base_triggers(), "safe_stop_entered": True}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_safe_stop_entry", ReviewMode.BLOCKING) in fired

    def test_evaluate_safe_stop_fires_supervisory(self):
        """safe_stop_entered=True fires under SUPERVISORY."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "safe_stop_entered": True}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_safe_stop_entry", ReviewMode.BLOCKING) in fired

    def test_evaluate_safe_stop_fires_mandatory(self):
        """safe_stop_entered=True fires under MANDATORY."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.MANDATORY)
        inputs = {**_base_triggers(), "safe_stop_entered": True}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_safe_stop_entry", ReviewMode.BLOCKING) in fired

    def test_evaluate_breaker_change_notify_supervisory(self):
        """breaker_mode_changed=True under SUPERVISORY → (on_breaker_mode_change, NOTIFY)."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "breaker_mode_changed": True}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_breaker_mode_change", ReviewMode.NOTIFY) in fired

    def test_evaluate_budget_low_threshold(self):
        """budget_remaining_fraction=0.08 under SUPERVISORY (threshold 0.10) → fires."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "budget_remaining_fraction": 0.08}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_budget_low", ReviewMode.NOTIFY) in fired

    def test_evaluate_budget_low_above_threshold(self):
        """budget_remaining_fraction=0.15 under SUPERVISORY (threshold 0.10) → does not fire."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "budget_remaining_fraction": 0.15}
        fired = t.evaluate_triggers(**inputs)
        trigger_names = [name for name, _ in fired]
        assert "on_budget_low" not in trigger_names

    def test_evaluate_multiple_triggers_fire(self):
        """REJECT + safe_stop_entered + budget_exhausted → all three in fired list."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {
            **_base_triggers(),
            "enforcement_result": "reject",
            "safe_stop_entered": True,
            "budget_exhausted": True,
        }
        fired = t.evaluate_triggers(**inputs)
        trigger_names = [name for name, _ in fired]
        assert "on_reject" in trigger_names
        assert "on_safe_stop_entry" in trigger_names
        assert "on_budget_exhaustion" in trigger_names

    def test_evaluate_returns_list(self):
        """evaluate_triggers always returns a list (never None)."""
        t = HumanReviewTriggers()
        result = t.evaluate_triggers(**_base_triggers())
        assert isinstance(result, list)

    def test_evaluate_trusted_field_mismatch_fires(self):
        """Trusted field mismatch above threshold fires the configured mode."""
        t = HumanReviewTriggers(
            on_trusted_field_large_mismatch_fraction=0.5,
            on_trusted_field_mismatch_mode=ReviewMode.BLOCKING,
        )
        inputs = {**_base_triggers(), "trusted_field_mismatch_fraction": 0.8}
        fired = t.evaluate_triggers(**inputs)
        assert ("on_trusted_field_mismatch", ReviewMode.BLOCKING) in fired

    def test_evaluate_trusted_field_mismatch_below_threshold(self):
        """Trusted field mismatch below threshold does not fire."""
        t = HumanReviewTriggers(
            on_trusted_field_large_mismatch_fraction=0.5,
            on_trusted_field_mismatch_mode=ReviewMode.BLOCKING,
        )
        inputs = {**_base_triggers(), "trusted_field_mismatch_fraction": 0.3}
        fired = t.evaluate_triggers(**inputs)
        trigger_names = [name for name, _ in fired]
        assert "on_trusted_field_mismatch" not in trigger_names

    def test_evaluate_safe_stop_suppresses_breaker_change(self):
        """When safe_stop_entered=True, breaker_mode_change trigger does not additionally fire."""
        t = HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
        inputs = {**_base_triggers(), "safe_stop_entered": True, "breaker_mode_changed": True}
        fired = t.evaluate_triggers(**inputs)
        trigger_names = [name for name, _ in fired]
        # safe_stop_entry fires, but breaker_mode_change is suppressed when safe_stop_entered
        assert "on_safe_stop_entry" in trigger_names
        assert "on_breaker_mode_change" not in trigger_names


# ═══════════════════════════════════════════════════════════════════════════
#  PRIORITY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestHighestPriorityTrigger:
    def test_highest_priority_blocking_over_notify(self):
        """BLOCKING trigger is returned over NOTIFY trigger regardless of list order."""
        t = HumanReviewTriggers()
        fired = [
            ("on_project_flagged", ReviewMode.NOTIFY),
            ("on_reject", ReviewMode.BLOCKING),
        ]
        result = t.highest_priority_trigger(fired)
        assert result == ("on_reject", ReviewMode.BLOCKING)

    def test_highest_priority_audit_over_reject(self):
        """audit_chain_failure outranks reject when both are BLOCKING."""
        t = HumanReviewTriggers()
        fired = [
            ("on_reject", ReviewMode.BLOCKING),
            ("on_audit_chain_failure", ReviewMode.BLOCKING),
        ]
        result = t.highest_priority_trigger(fired)
        assert result is not None
        assert result[0] == "on_audit_chain_failure"

    def test_highest_priority_empty_list(self):
        """Empty fired list returns None."""
        t = HumanReviewTriggers()
        assert t.highest_priority_trigger([]) is None

    def test_highest_priority_single_item(self):
        """Single-item list returns that item."""
        t = HumanReviewTriggers()
        fired = [("on_reject", ReviewMode.NOTIFY)]
        result = t.highest_priority_trigger(fired)
        assert result == ("on_reject", ReviewMode.NOTIFY)

    def test_highest_priority_safe_stop_over_reject(self):
        """safe_stop_entry outranks reject when both are BLOCKING."""
        t = HumanReviewTriggers()
        fired = [
            ("on_reject", ReviewMode.BLOCKING),
            ("on_safe_stop_entry", ReviewMode.BLOCKING),
        ]
        result = t.highest_priority_trigger(fired)
        assert result is not None
        assert result[0] == "on_safe_stop_entry"

    def test_highest_priority_notify_when_no_blocking(self):
        """When no BLOCKING trigger fires, the highest-priority NOTIFY is returned."""
        t = HumanReviewTriggers()
        fired = [
            ("on_project_flagged", ReviewMode.NOTIFY),
            ("on_budget_low", ReviewMode.NOTIFY),
        ]
        result = t.highest_priority_trigger(fired)
        assert result is not None
        assert result[0] == "on_project_flagged"  # higher priority than budget_low


# ═══════════════════════════════════════════════════════════════════════════
#  GATEWAY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLocalApprovalGateway:
    def test_submit_returns_review_id(self):
        """First submit returns 'review-0'."""
        gw = LocalApprovalGateway()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        assert rid == "review-0"

    def test_submit_increments_counter(self):
        """Second submit returns 'review-1'."""
        gw = LocalApprovalGateway()
        rid0 = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        rid1 = gw.submit_for_review(_FakeOutput(), "on_safe_stop_entry", {})
        assert rid0 == "review-0"
        assert rid1 == "review-1"

    def test_poll_no_decision(self):
        """poll_decision returns None when no decision has been programmed."""
        gw = LocalApprovalGateway()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        assert gw.poll_decision(rid) is None

    def test_poll_with_programmed_decision(self):
        """poll_decision returns the programmed decision."""
        gw = LocalApprovalGateway()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        d = _decision(review_id=rid, action=HumanDecisionAction.APPROVE)
        gw.program_decision(rid, d)
        result = gw.poll_decision(rid)
        assert result is not None
        assert result.action == HumanDecisionAction.APPROVE
        assert result.review_id == rid

    def test_poll_removes_decision(self):
        """Decision is consumed — second poll returns None."""
        gw = LocalApprovalGateway()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        gw.program_decision(rid, _decision(review_id=rid))
        gw.poll_decision(rid)
        assert gw.poll_decision(rid) is None

    def test_cancel_review(self):
        """cancel_review removes the submission from pending."""
        gw = LocalApprovalGateway()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        assert gw.get_pending_count() == 1
        gw.cancel_review(rid)
        assert gw.get_pending_count() == 0

    def test_auto_approve_all(self):
        """set_auto_approve: submit then poll yields an APPROVE decision."""
        gw = LocalApprovalGateway()
        gw.set_auto_approve(reviewer_id="ci-bot")
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        decision = gw.poll_decision(rid)
        assert decision is not None
        assert decision.action == HumanDecisionAction.APPROVE
        assert decision.reviewer_id == "ci-bot"
        assert decision.authenticated is True

    def test_auto_deny_all(self):
        """set_auto_deny: submit then poll yields a DENY decision."""
        gw = LocalApprovalGateway()
        gw.set_auto_deny(reviewer_id="ci-denier")
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        decision = gw.poll_decision(rid)
        assert decision is not None
        assert decision.action == HumanDecisionAction.DENY
        assert decision.reviewer_id == "ci-denier"
        assert decision.authenticated is True

    def test_set_manual_clears_auto(self):
        """After set_manual(), submit no longer creates auto decisions."""
        gw = LocalApprovalGateway()
        gw.set_auto_approve()
        gw.set_manual()
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        assert gw.poll_decision(rid) is None

    def test_cancel_nonexistent_review_is_safe(self):
        """cancel_review on an unknown ID does not raise."""
        gw = LocalApprovalGateway()
        gw.cancel_review("review-99")  # should not raise

    def test_pending_count_tracks_submissions(self):
        """get_pending_count reflects open reviews accurately."""
        gw = LocalApprovalGateway()
        assert gw.get_pending_count() == 0
        rid0 = gw.submit_for_review(_FakeOutput(), "on_reject", {})
        assert gw.get_pending_count() == 1
        gw.submit_for_review(_FakeOutput(), "on_safe_stop_entry", {})
        assert gw.get_pending_count() == 2
        gw.cancel_review(rid0)
        assert gw.get_pending_count() == 1

    def test_context_is_stored_in_pending(self):
        """submit_for_review stores context and the gateway tracks it internally."""
        gw = LocalApprovalGateway()
        ctx = {"breaker_mode": "THROTTLED", "policy_version": "v2"}
        rid = gw.submit_for_review(_FakeOutput(), "on_reject", ctx)
        assert gw.get_pending_count() == 1
        # The submission was stored; consuming the decision clears it
        gw.program_decision(rid, _decision(review_id=rid))
        gw.poll_decision(rid)
        assert gw.get_pending_count() == 0

    def test_gateway_satisfies_protocol(self):
        """LocalApprovalGateway is structurally compatible with ApprovalGateway."""
        # Protocol compliance: the three required methods are present and callable
        gw = LocalApprovalGateway()
        assert callable(gw.submit_for_review)
        assert callable(gw.poll_decision)
        assert callable(gw.cancel_review)
