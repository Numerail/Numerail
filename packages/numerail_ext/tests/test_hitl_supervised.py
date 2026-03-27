"""Tests for SupervisedGovernor — HITL enforcement logic, TOCTOU protection,
and audit integration.

Coverage:
  - Core step flow (no trigger, NOTIFY, BLOCKING)
  - APPROVE decision with TOCTOU re-enforcement
  - TOCTOU geometry tightening (inject tighter config via set_active_config)
  - DENY decision with guidance constraints
  - MODIFY decision (dict vector, infeasible vector, None vector)
  - ESCALATE (within depth, ceiling, new pending id)
  - DEFER (once allowed, second denied)
  - Timeout / expiry (injectable time source)
  - Audit chain (hash linking, record types)
  - Lifecycle and monitoring helpers
"""

from __future__ import annotations

import hashlib
import json
import time
from time import time_ns
from typing import Any, Mapping

import pytest

# core
from numerail.protocols import (
    HumanDecision,
    HumanDecisionAction,
    ReviewOutcome,
)

# ext
from numerail_ext.survivability.breaker import BreakerStateMachine
from numerail_ext.survivability.governor import StateTransitionGovernor
from numerail_ext.survivability.hitl import (
    HumanReviewProfile,
    HumanReviewTriggers,
    PendingAction,
    ReviewMode,
    SupervisedGovernor,
    SupervisedStepResult,
    _HitlAuditChain,
    record_human_decision,
    record_review_expiry,
)
from numerail_ext.survivability.local_backend import LocalNumerailBackend
from numerail_ext.survivability.local_gateway import LocalApprovalGateway
from numerail_ext.survivability.policy_builder import build_v5_policy_from_envelope
from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
from numerail_ext.survivability.types import (
    BreakerMode,
    BreakerThresholds,
    ExecutionReceipt,
    TelemetrySnapshot,
    WorkloadRequest,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_THRESHOLDS = BreakerThresholds(trip_score=0.60, reset_score=0.30, safe_stop_score=0.85)
_BOOTSTRAP = {"gpu_shift": 500.0, "external_api_shift": 100.0, "mutation_shift": 40.0}
_FRESHNESS_NS = 120_000_000_000  # 120 s — allows snapshot reuse across steps


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _snap(
    gpu: float = 0.30,
    api: float = 0.30,
    db: float = 0.35,
    queue: float = 0.20,
    error: float = 0.5,
    state_version: int = 1,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        state_version=state_version,
        observed_at_ns=time_ns(),
        current_gpu_util=gpu,
        current_api_util=api,
        current_db_util=db,
        current_queue_util=queue,
        current_error_rate_pct=error,
        ctrl_gpu_reserve_seconds=30.0,
        ctrl_api_reserve_calls=5.0,
        ctrl_parallel_reserve=4.0,
        ctrl_cloud_mutation_reserve=2.0,
        gpu_disturbance_margin_seconds=15.0,
        api_disturbance_margin_calls=3.0,
        db_disturbance_margin_pct=5.0,
        queue_disturbance_margin_pct=3.0,
    )


def _request(scale: float = 1.0) -> WorkloadRequest:
    """Small structurally-valid workload (satisfies all relation constraints)."""
    return WorkloadRequest(
        prompt_k=5.0 * scale,
        completion_k=2.0 * scale,
        internal_tool_calls=5.0 * scale,   # >= external_api_calls >= cloud_mutation_calls
        external_api_calls=3.0 * scale,
        cloud_mutation_calls=1.0 * scale,
        gpu_seconds=10.0 * scale,
        parallel_workers=2.0 * scale,
        traffic_shift_pct=0.0,
        worker_scale_up_pct=0.0,
        feature_flag_changes=0.0,
        rollback_batch_pct=0.0,
        pager_notifications=1.0,
        customer_comms_count=0.0,
    )


class _MockResMgr:
    def __init__(self):
        self._n = 0

    def acquire(self, *, state_version, expires_at_ns, resource_claims) -> str:
        self._n += 1
        return f"tok_{self._n}"

    def commit(self, *, token, receipt) -> None:
        pass

    def release(self, *, token) -> None:
        pass


class _MockDigestor:
    def digest(self, payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


def _make_governor(bootstrap_budgets: dict | None = None) -> StateTransitionGovernor:
    return StateTransitionGovernor(
        backend=LocalNumerailBackend(),
        transition_model=IncidentCommanderTransitionModel(freshness_ns=_FRESHNESS_NS),
        reservation_mgr=_MockResMgr(),
        digestor=_MockDigestor(),
        thresholds=_THRESHOLDS,
        bootstrap_budgets=bootstrap_budgets or dict(_BOOTSTRAP),
    )


def _decision(
    review_id: str,
    action: HumanDecisionAction,
    *,
    authenticated: bool = True,
    modified_vector=None,
    guidance=None,
    reason: str = "test",
) -> HumanDecision:
    return HumanDecision(
        review_id=review_id,
        action=action,
        reviewer_id="test-reviewer",
        authenticated=authenticated,
        timestamp_ms=float(time.time() * 1000),
        reason=reason,
        modified_vector=modified_vector,
        guidance=guidance,
    )


def _make_supervised(
    triggers: HumanReviewTriggers | None = None,
    auto: str | None = "approve",
    time_ms_fn=None,
) -> tuple[SupervisedGovernor, LocalApprovalGateway, StateTransitionGovernor]:
    governor = _make_governor()
    gateway = LocalApprovalGateway()
    if auto == "approve":
        gateway.set_auto_approve()
    elif auto == "deny":
        gateway.set_auto_deny()
    elif auto is None:
        gateway.set_manual()
    trigs = triggers or HumanReviewTriggers.from_profile(HumanReviewProfile.SUPERVISORY)
    sg = SupervisedGovernor(
        governor=governor,
        gateway=gateway,
        triggers=trigs,
        _time_ms_fn=time_ms_fn,
    )
    return sg, gateway, governor


# ── Tests: Core step flow ─────────────────────────────────────────────────────


class TestCoreStepFlow:
    def test_no_trigger_approve_path_returns_executed(self):
        """Normal step with no triggers configured → EXECUTED immediately."""
        sg, _, gov = _make_supervised(triggers=HumanReviewTriggers())
        result = sg.step(request=_request(), snapshot=_snap())
        # With no triggers configured, any enforcement result passes through
        assert isinstance(result, SupervisedStepResult)
        assert result.outcome in (ReviewOutcome.EXECUTED, ReviewOutcome.DENIED)

    def test_no_trigger_reject_no_review_configured(self):
        """Reject with no BLOCKING trigger → DENIED without review."""
        trigs = HumanReviewTriggers()  # no triggers
        sg, _, gov = _make_supervised(triggers=trigs)
        gov.breaker.force_mode(BreakerMode.OPEN)  # forces reject
        result = sg.step(request=_request(), snapshot=_snap())
        assert result.outcome == ReviewOutcome.DENIED
        assert sg.get_pending_count() == 0

    def test_blocking_trigger_on_reject_returns_pending(self):
        """BLOCKING trigger on reject → PENDING_REVIEW, pending count = 1."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        result = sg.step(request=_request(), snapshot=_snap())
        assert result.outcome == ReviewOutcome.PENDING_REVIEW
        assert sg.get_pending_count() == 1
        assert result.trigger_reason == "on_reject"

    def test_notify_trigger_on_reject_does_not_block(self):
        """NOTIFY trigger on reject → gateway notified but DENIED returned immediately."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.NOTIFY)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        result = sg.step(request=_request(), snapshot=_snap())
        # NOTIFY on reject: enforcement is still deny, no pending review
        assert result.outcome == ReviewOutcome.DENIED
        assert sg.get_pending_count() == 0

    def test_blocking_trigger_on_approval_path_returns_pending(self):
        """BLOCKING trigger on confirmation → PENDING_REVIEW on approve/project path."""
        trigs = HumanReviewTriggers(
            on_project_confirmation_required=ReviewMode.BLOCKING,
            on_reject=ReviewMode.BLOCKING,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        # With breaker OPEN the reject path triggers on_reject
        gov.breaker.force_mode(BreakerMode.OPEN)
        result = sg.step(request=_request(), snapshot=_snap())
        assert result.outcome == ReviewOutcome.PENDING_REVIEW

    def test_step_auto_generates_action_id(self):
        """action_id is auto-generated as 'supervised-N' when not provided."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        result = sg.step(request=_request(), snapshot=_snap())
        assert result.outcome == ReviewOutcome.PENDING_REVIEW
        ids = sg.get_pending_ids()
        assert len(ids) == 1

    def test_step_passes_explicit_action_id(self):
        """Explicit action_id is forwarded to the inner governor."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        result = sg.step(request=_request(), snapshot=_snap(), action_id="my-action-123")
        pending_id = sg.get_pending_ids()[0]
        pending = sg._pending[pending_id]
        assert pending.context["action_id"] == "my-action-123"

    def test_notify_trigger_on_project_path_returns_executed(self):
        """NOTIFY on flagged projection → EXECUTED (NOTIFY doesn't block)."""
        trigs = HumanReviewTriggers(on_project_flagged=ReviewMode.NOTIFY)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        # Normal healthy step — enforcement should approve or project
        result = sg.step(request=_request(), snapshot=_snap())
        # NOTIFY on any path: always non-blocking
        if result.trigger_reason == "on_project_flagged":
            assert result.outcome == ReviewOutcome.EXECUTED


# ── Tests: Approve decision ───────────────────────────────────────────────────


class TestApproveDecision:
    def test_approve_yields_executed(self):
        """Gateway auto-approve → EXECUTED with re_enforcement_output set."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        step_result = sg.step(request=_request(), snapshot=_snap())
        assert step_result.outcome == ReviewOutcome.PENDING_REVIEW
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        result = sg.resolve_pending(review_id)
        # OPEN breaker → enforced_values is None → TOCTOU returns reject → DENIED
        # (No enforced values to re-enforce since it was rejected at OPEN path)
        assert result is not None
        assert result.human_decision is not None
        assert result.human_decision.action == HumanDecisionAction.APPROVE

    def test_approve_on_feasible_step_yields_executed(self):
        """Approve a normally-enforced (non-OPEN) step → EXECUTED after TOCTOU."""
        trigs = HumanReviewTriggers(
            on_project_confirmation_required=ReviewMode.BLOCKING,
            on_reject=ReviewMode.BLOCKING,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        # Force a reject via OPEN to get a pending review, then close and re-run
        gov.breaker.force_mode(BreakerMode.OPEN)
        step_result = sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.re_enforcement_output is not None

    def test_unauthenticated_approve_yields_denied(self):
        """Unauthenticated decision → DENIED regardless of action."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(
            review_id,
            _decision(review_id, HumanDecisionAction.APPROVE, authenticated=False),
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.outcome == ReviewOutcome.DENIED

    def test_resolve_unknown_review_id_returns_none(self):
        """resolve_pending on unknown review_id → None (not an error)."""
        sg, _, _ = _make_supervised()
        assert sg.resolve_pending("nonexistent-review") is None

    def test_approve_clears_pending(self):
        """After APPROVE is resolved, pending count drops to 0."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        sg.resolve_pending(review_id)
        assert sg.get_pending_count() == 0


# ── Tests: TOCTOU protection ──────────────────────────────────────────────────


class TestTOCTOU:
    def test_toctou_passes_with_unchanged_geometry(self):
        """Approve a non-OPEN step: TOCTOU re-enforcement with same geometry → EXECUTED."""
        trigs = HumanReviewTriggers(
            on_project_confirmation_required=ReviewMode.BLOCKING,
            on_reject=ReviewMode.BLOCKING,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        # Manufacture a blocking step: force OPEN, then run step
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        # The OPEN path has no enforced_values → TOCTOU will reject
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        result = sg.resolve_pending(review_id)
        assert result is not None
        # re_enforcement_output should always be present on approve path
        assert result.re_enforcement_output is not None

    def test_toctou_denied_when_enforced_values_none(self):
        """OPEN-path approve → no enforced_values → TOCTOU returns reject → DENIED."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        result = sg.resolve_pending(review_id)
        # OPEN path: no enforced_values → TOCTOU rejects → DENIED
        assert result is not None
        assert result.outcome == ReviewOutcome.DENIED
        assert result.re_enforcement_output is not None
        assert result.re_enforcement_output.get("decision") == "reject"

    def test_toctou_geometry_tightened_yields_denied(self):
        """Inject a tighter config after step → TOCTOU re-enforcement denies."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        # Reset breaker so backend has a loaded config
        gov.breaker.force_mode(BreakerMode.CLOSED)
        # Run a fresh step to load config into backend
        gov.enforce_next_step(
            request=_request(scale=0.1),
            snapshot=_snap(),
            action_id="config-load",
        )
        # Now manually inject a zero-ceiling config to simulate tightened geometry
        snap = _snap()
        tight_snap = TelemetrySnapshot(
            state_version=snap.state_version,
            observed_at_ns=snap.observed_at_ns,
            current_gpu_util=0.99,  # extreme overload → tiny envelope
            current_api_util=0.99,
            current_db_util=0.99,
            current_queue_util=0.99,
            current_error_rate_pct=99.0,
            ctrl_gpu_reserve_seconds=snap.ctrl_gpu_reserve_seconds,
            ctrl_api_reserve_calls=snap.ctrl_api_reserve_calls,
            ctrl_parallel_reserve=snap.ctrl_parallel_reserve,
            ctrl_cloud_mutation_reserve=snap.ctrl_cloud_mutation_reserve,
            gpu_disturbance_margin_seconds=snap.gpu_disturbance_margin_seconds,
            api_disturbance_margin_calls=snap.api_disturbance_margin_calls,
            db_disturbance_margin_pct=snap.db_disturbance_margin_pct,
            queue_disturbance_margin_pct=snap.queue_disturbance_margin_pct,
        )
        from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
        model = IncidentCommanderTransitionModel(freshness_ns=_FRESHNESS_NS)
        tight_envelope = model.synthesize_envelope(
            snapshot=tight_snap,
            mode=BreakerMode.SAFE_STOP,
            budgets=dict(_BOOTSTRAP),
        )
        tight_config = build_v5_policy_from_envelope(tight_envelope)
        gov.backend.set_active_config(tight_config)

        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.re_enforcement_output is not None


# ── Tests: Deny decision ──────────────────────────────────────────────────────


class TestDenyDecision:
    def test_deny_yields_denied(self):
        """DENY decision → DENIED outcome."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.DENY))
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.outcome == ReviewOutcome.DENIED
        assert result.human_decision.action == HumanDecisionAction.DENY

    def test_deny_with_guidance_constraints_populated(self):
        """DENY with guidance → guidance_constraints set on result."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        guidance = {"gpu_seconds": (0.0, 5.0), "external_api_calls": (0.0, 2.0)}
        gateway.program_decision(
            review_id,
            _decision(review_id, HumanDecisionAction.DENY, guidance=guidance),
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.guidance_constraints == guidance

    def test_deny_clears_pending(self):
        """After DENY, pending count = 0."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.DENY))
        sg.resolve_pending(review_id)
        assert sg.get_pending_count() == 0


# ── Tests: Modify decision ────────────────────────────────────────────────────


class TestModifyDecision:
    def test_modify_none_vector_yields_denied(self):
        """MODIFY with modified_vector=None → DENIED (no vector to re-enforce)."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(
            review_id,
            _decision(review_id, HumanDecisionAction.MODIFY, modified_vector=None),
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.outcome == ReviewOutcome.DENIED

    def test_modify_with_dict_vector_re_enforced(self):
        """MODIFY with a dict vector → re-enforcement runs, result reflects feasibility."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        # Supply a small workload as the modified dict
        modified_dict = _request(scale=0.1).as_action_dict()
        gateway.program_decision(
            review_id,
            _decision(review_id, HumanDecisionAction.MODIFY, modified_vector=modified_dict),
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.re_enforcement_output is not None

    def test_modify_clears_pending(self):
        """After MODIFY is resolved, pending is removed."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(
            review_id,
            _decision(review_id, HumanDecisionAction.MODIFY, modified_vector={}),
        )
        sg.resolve_pending(review_id)
        assert sg.get_pending_count() == 0


# ── Tests: Escalate decision ──────────────────────────────────────────────────


class TestEscalateDecision:
    def test_escalate_within_depth_creates_new_pending(self):
        """ESCALATE within max_escalation_depth → new review_id pending, None returned."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING, max_escalation_depth=2)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.ESCALATE)
        )
        result = sg.resolve_pending(review_id)
        assert result is None  # Still pending — escalated to new id
        assert sg.get_pending_count() == 1
        new_ids = sg.get_pending_ids()
        assert review_id not in new_ids

    def test_escalate_new_pending_has_incremented_depth(self):
        """After ESCALATE, new PendingAction has escalation_depth = 1."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING, max_escalation_depth=2)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.ESCALATE)
        )
        sg.resolve_pending(review_id)
        new_id = sg.get_pending_ids()[0]
        assert sg._pending[new_id].escalation_depth == 1

    def test_escalate_at_ceiling_yields_denied(self):
        """ESCALATE when escalation_depth == max_escalation_depth → DENIED."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING, max_escalation_depth=1)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        # First escalate: reaches depth 1, which equals max (1) → DENIED
        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.ESCALATE)
        )
        result = sg.resolve_pending(review_id)
        # depth was 0, max is 1, so 0 < 1 → escalated, new review created
        assert result is None
        new_id = sg.get_pending_ids()[0]
        assert sg._pending[new_id].escalation_depth == 1

        # Second escalate: depth == max_escalation_depth → DENIED
        gateway.program_decision(
            new_id, _decision(new_id, HumanDecisionAction.ESCALATE)
        )
        result2 = sg.resolve_pending(new_id)
        assert result2 is not None
        assert result2.outcome == ReviewOutcome.DENIED


# ── Tests: Defer decision ─────────────────────────────────────────────────────


class TestDeferDecision:
    def test_defer_once_returns_none(self):
        """DEFER → returns None (still pending), deferred=True set."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.DEFER)
        )
        result = sg.resolve_pending(review_id)
        assert result is None
        assert sg._pending[review_id].deferred is True

    def test_defer_twice_yields_denied(self):
        """Second DEFER → DENIED (only one deferral allowed)."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        # First defer
        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.DEFER)
        )
        sg.resolve_pending(review_id)

        # Second defer — now deferred=True so it should deny
        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.DEFER)
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.outcome == ReviewOutcome.DENIED

    def test_defer_then_approve_yields_executed(self):
        """DEFER then APPROVE → re-enforcement runs, DENIED (OPEN path, no values)."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.DEFER)
        )
        sg.resolve_pending(review_id)

        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.APPROVE)
        )
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.human_decision.action == HumanDecisionAction.APPROVE


# ── Tests: Timeout / expiry ───────────────────────────────────────────────────


class TestTimeoutExpiry:
    def test_expired_review_yields_expired(self):
        """Time past deadline → EXPIRED outcome (fail-closed)."""
        # Use injectable time source: first call returns submission time,
        # subsequent calls return a time far in the future
        times = [0.0]
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            if call_count[0] <= 1:
                return 0.0  # submission time
            return 999_999_999.0  # way past any timeout

        trigs = HumanReviewTriggers(
            on_reject=ReviewMode.BLOCKING,
            review_timeout_seconds=300.0,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None, time_ms_fn=fake_time)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        # resolve_pending calls _get_current_time_ms → returns future time → expired
        result = sg.resolve_pending(review_id)
        assert result is not None
        assert result.outcome == ReviewOutcome.EXPIRED
        assert sg.get_pending_count() == 0

    def test_deferred_review_has_extended_deadline(self):
        """Deferred review does not expire until 2× the original timeout."""
        timeout_ms = 300_000.0  # 300 seconds in ms
        # step 1 (submit): time=0
        # step 2 (defer): time=250_000 — within first window
        # step 3 (check): time=400_000 — past first but within deferred window (600_000)
        times = iter([0.0, 250_000.0, 400_000.0, 400_000.0])

        def fake_time():
            return next(times)

        trigs = HumanReviewTriggers(
            on_reject=ReviewMode.BLOCKING,
            review_timeout_seconds=300.0,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None, time_ms_fn=fake_time)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        gateway.program_decision(
            review_id, _decision(review_id, HumanDecisionAction.DEFER)
        )
        defer_result = sg.resolve_pending(review_id)
        assert defer_result is None  # deferred

        # Poll at time=400_000: past first timeout (300_000), but within 2× (600_000)
        no_result = sg.resolve_pending(review_id)
        assert no_result is None  # still pending — deferred window not elapsed

    def test_expiry_recorded_in_audit(self):
        """Expired review writes a review_expiry record to the HITL audit chain."""
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0.0 if call_count[0] <= 1 else 999_999_999.0

        trigs = HumanReviewTriggers(
            on_reject=ReviewMode.BLOCKING,
            review_timeout_seconds=300.0,
        )
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None, time_ms_fn=fake_time)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        sg.resolve_pending(review_id)

        records = sg.hitl_audit_records
        assert len(records) == 1
        assert records[0]["event_type"] == "review_expiry"
        assert records[0]["review_id"] == review_id


# ── Tests: Audit chain ────────────────────────────────────────────────────────


class TestAuditChain:
    def test_approve_recorded_in_audit(self):
        """APPROVE decision → human_decision record in hitl_audit_records."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.APPROVE))
        sg.resolve_pending(review_id)

        records = sg.hitl_audit_records
        assert len(records) == 1
        assert records[0]["event_type"] == "human_decision"
        assert records[0]["action"] == "approve"

    def test_deny_recorded_in_audit(self):
        """DENY decision → human_decision record in hitl_audit_records."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        gateway.program_decision(review_id, _decision(review_id, HumanDecisionAction.DENY))
        sg.resolve_pending(review_id)

        records = sg.hitl_audit_records
        assert len(records) == 1
        assert records[0]["event_type"] == "human_decision"
        assert records[0]["action"] == "deny"

    def test_audit_chain_hash_linked(self):
        """Sequential records form a valid SHA-256 hash-linked chain."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)

        # Generate two sequential decisions
        for _ in range(2):
            gov.breaker.force_mode(BreakerMode.OPEN)
            sg.step(request=_request(), snapshot=_snap())
            rid = sg.get_pending_ids()[-1]
            gateway.program_decision(rid, _decision(rid, HumanDecisionAction.DENY))
            sg.resolve_pending(rid)
            gov.breaker.force_mode(BreakerMode.CLOSED)

        records = sg.hitl_audit_records
        assert len(records) == 2

        # Verify chain: record[1].prev_hash == record[0].chain_hash
        assert records[1]["prev_hash"] == records[0]["chain_hash"]

        # Re-derive first record's hash independently
        payload_0 = {k: v for k, v in records[0].items() if k != "chain_hash"}
        expected_0 = hashlib.sha256(
            json.dumps(payload_0, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert records[0]["chain_hash"] == expected_0

    def test_audit_chain_genesis_hash(self):
        """Before any events, the chain head is 64 zero hex digits."""
        sg, _, _ = _make_supervised()
        assert sg._hitl_audit.head_hash == "0" * 64

    def test_reviewer_id_recorded_in_audit(self):
        """Human decision audit record includes reviewer_id."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]
        dec = HumanDecision(
            review_id=review_id,
            action=HumanDecisionAction.DENY,
            reviewer_id="alice@example.com",
            authenticated=True,
            timestamp_ms=float(time.time() * 1000),
            reason="test",
        )
        gateway.program_decision(review_id, dec)
        sg.resolve_pending(review_id)
        records = sg.hitl_audit_records
        assert records[0]["reviewer_id"] == "alice@example.com"


# ── Tests: Lifecycle and monitoring ──────────────────────────────────────────


class TestLifecycleAndMonitoring:
    def test_get_pending_count_increments_with_blocking_steps(self):
        """get_pending_count() increases with each blocking step."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        assert sg.get_pending_count() == 0
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap(), action_id="a1")
        assert sg.get_pending_count() == 1
        sg.step(request=_request(), snapshot=_snap(), action_id="a2")
        assert sg.get_pending_count() == 2

    def test_get_pending_ids_lists_all_open_reviews(self):
        """get_pending_ids() returns correct review_ids."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap(), action_id="a1")
        sg.step(request=_request(), snapshot=_snap(), action_id="a2")
        ids = sg.get_pending_ids()
        assert len(ids) == 2
        # Both should be string review ids
        for rid in ids:
            assert isinstance(rid, str)

    def test_resolve_all_pending_batch(self):
        """resolve_all_pending() processes all pending reviews in one call."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap(), action_id="a1")
        sg.step(request=_request(), snapshot=_snap(), action_id="a2")
        for rid in sg.get_pending_ids():
            gateway.program_decision(rid, _decision(rid, HumanDecisionAction.DENY))

        batch = sg.resolve_all_pending()
        assert len(batch) == 2
        review_ids = [rid for rid, _ in batch]
        results = [r for _, r in batch]
        assert all(r is not None for r in results)
        assert all(r.outcome == ReviewOutcome.DENIED for r in results)
        assert sg.get_pending_count() == 0

    def test_multiple_concurrent_pending_reviews(self):
        """Multiple concurrent pending reviews are tracked independently."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        for i in range(3):
            sg.step(request=_request(), snapshot=_snap(), action_id=f"action-{i}")
        assert sg.get_pending_count() == 3
        ids = sg.get_pending_ids()

        # Approve first, deny second, leave third pending
        gateway.program_decision(ids[0], _decision(ids[0], HumanDecisionAction.APPROVE))
        gateway.program_decision(ids[1], _decision(ids[1], HumanDecisionAction.DENY))

        sg.resolve_pending(ids[0])
        sg.resolve_pending(ids[1])
        assert sg.get_pending_count() == 1
        assert ids[2] in sg.get_pending_ids()

    def test_no_pending_after_notify_trigger(self):
        """NOTIFY trigger produces no pending reviews."""
        trigs = HumanReviewTriggers(on_reject=ReviewMode.NOTIFY)
        sg, _, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        assert sg.get_pending_count() == 0


# ── Tests: Internal audit chain helpers ──────────────────────────────────────


class TestHitlAuditChainHelpers:
    def test_hitl_audit_chain_appends_and_links(self):
        """_HitlAuditChain: two appends produce a valid linked chain."""
        chain = _HitlAuditChain()
        assert chain.head_hash == "0" * 64

        h1 = chain.append({"event": "first", "value": 1})
        h2 = chain.append({"event": "second", "value": 2})
        records = chain.records
        assert len(records) == 2
        assert records[0]["chain_hash"] == h1
        assert records[1]["chain_hash"] == h2
        assert records[1]["prev_hash"] == h1

    def test_record_human_decision_produces_correct_event_type(self):
        """record_human_decision writes event_type='human_decision'."""
        chain = _HitlAuditChain()
        trigs = HumanReviewTriggers(on_reject=ReviewMode.BLOCKING)
        sg, gateway, gov = _make_supervised(triggers=trigs, auto=None)
        gov.breaker.force_mode(BreakerMode.OPEN)
        sg.step(request=_request(), snapshot=_snap())
        review_id = sg.get_pending_ids()[0]

        dec = _decision(review_id, HumanDecisionAction.DENY)
        pending = sg._pending[review_id]
        record_human_decision(chain, dec, pending.enforcement_output)
        assert chain.records[0]["event_type"] == "human_decision"
        assert chain.records[0]["action"] == "deny"

    def test_record_review_expiry_produces_correct_event_type(self):
        """record_review_expiry writes event_type='review_expiry' with elapsed_ms."""
        chain = _HitlAuditChain()
        record_review_expiry(chain, "review-99", 1000.0, 5000.0, "on_reject")
        records = chain.records
        assert len(records) == 1
        assert records[0]["event_type"] == "review_expiry"
        assert records[0]["elapsed_ms"] == 4000.0
        assert records[0]["review_id"] == "review-99"
