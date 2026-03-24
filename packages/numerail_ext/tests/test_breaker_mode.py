"""Exhaustive tests for the breaker mode survivability extension.

Covers: breaker state machine, transition model, policy builder → V5 integration,
validation, governor lifecycle, local backend, guarantee fuzzing.

Governor and fuzz tests use ``freshness_ns=120_000_000_000`` (120 seconds) so that
the freshness window does not expire during the 30-dimensional mixed-constraint
solver pass in the test environment.  In production the default 5-second window
is appropriate because enforcement runs against a pre-loaded, pre-compiled policy
in single-digit milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import threading
from time import time_ns
from typing import Any, Mapping, Optional

import numpy as np
import pytest

from numerail.engine import EnforcementResult, NumerailSystem
from numerail.parser import PolicyParser, lint_config

from numerail_ext.survivability.types import (
    BreakerDecision,
    BreakerMode,
    BreakerThresholds,
    ExecutableGrant,
    ExecutionReceipt,
    GovernedStep,
    TelemetrySnapshot,
    TransitionEnvelope,
    WorkloadRequest,
)
from numerail_ext.survivability.breaker import BreakerStateMachine
from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
from numerail_ext.survivability.policy_builder import build_v5_policy_from_envelope
from numerail_ext.survivability.validation import (
    ReceiptValidationError,
    validate_receipt_against_grant,
)
from numerail_ext.survivability.governor import StateTransitionGovernor
from numerail_ext.survivability.local_backend import LocalNumerailBackend


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

# Extended freshness for test environment.  Prevents expiry during slow
# solver passes on 30-dim mixed-constraint policies.
_TEST_FRESHNESS_NS = 120_000_000_000


def _snap(
    gpu=0.45, api=0.30, db=0.40, queue=0.25, error=1.0,
    state_version=42, observed_at_ns=None,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        state_version=state_version,
        observed_at_ns=observed_at_ns or time_ns(),
        current_gpu_util=gpu, current_api_util=api,
        current_db_util=db, current_queue_util=queue,
        current_error_rate_pct=error,
        ctrl_gpu_reserve_seconds=30.0, ctrl_api_reserve_calls=5.0,
        ctrl_parallel_reserve=4.0, ctrl_cloud_mutation_reserve=2.0,
        gpu_disturbance_margin_seconds=15.0, api_disturbance_margin_calls=3.0,
        db_disturbance_margin_pct=5.0, queue_disturbance_margin_pct=3.0,
    )


def _small_request() -> WorkloadRequest:
    return WorkloadRequest(
        prompt_k=5.0, completion_k=2.0, internal_tool_calls=5.0,
        external_api_calls=3.0, cloud_mutation_calls=1.0, gpu_seconds=10.0,
        parallel_workers=2.0, traffic_shift_pct=2.0, worker_scale_up_pct=2.0,
        feature_flag_changes=1.0, rollback_batch_pct=1.0,
        pager_notifications=1.0, customer_comms_count=0.0,
    )


def _safe_stop_request() -> WorkloadRequest:
    return WorkloadRequest(
        prompt_k=2.0, completion_k=1.0, internal_tool_calls=3.0,
        external_api_calls=2.0, cloud_mutation_calls=1.0, gpu_seconds=5.0,
        parallel_workers=1.0, traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
        feature_flag_changes=0.0, rollback_batch_pct=0.0,
        pager_notifications=1.0, customer_comms_count=0.0,
    )


def _thresholds() -> BreakerThresholds:
    return BreakerThresholds(trip_score=0.60, reset_score=0.30, safe_stop_score=0.85)


def _budgets() -> dict[str, float]:
    return {"gpu_shift": 500.0, "external_api_shift": 100.0, "mutation_shift": 40.0}


def _test_model() -> IncidentCommanderTransitionModel:
    return IncidentCommanderTransitionModel(freshness_ns=_TEST_FRESHNESS_NS)


class _MockReservationMgr:
    def __init__(self):
        self.acquired = []
        self.committed = []
        self.released = []
        self._counter = 0

    def acquire(self, *, state_version, expires_at_ns, resource_claims):
        self._counter += 1
        token = f"tok_{self._counter}"
        self.acquired.append(token)
        return token

    def commit(self, *, token, receipt):
        self.committed.append(token)

    def release(self, *, token):
        self.released.append(token)


class _MockDigestor:
    def digest(self, payload):
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  1. BREAKER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class TestBreakerStateMachine:

    def test_initial_state_closed(self):
        assert BreakerStateMachine(_thresholds()).mode == BreakerMode.CLOSED

    def test_normal_stays_closed(self):
        b = BreakerStateMachine(_thresholds())
        d = b.update(_snap(gpu=0.2, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.CLOSED
        assert d.reason == "normal"

    def test_trip_to_throttled(self):
        b = BreakerStateMachine(_thresholds())
        d = b.update(_snap(gpu=0.9, api=0.8, db=0.7, queue=0.5, error=5.0))
        assert d.mode == BreakerMode.THROTTLED

    def test_throttled_to_half_open(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.THROTTLED)
        d = b.update(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.HALF_OPEN

    def test_half_open_to_closed(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.HALF_OPEN)
        d = b.update(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.CLOSED

    def test_half_open_retrip(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.HALF_OPEN)
        d = b.update(_snap(gpu=0.9, api=0.8, db=0.7, queue=0.5, error=5.0))
        assert d.mode == BreakerMode.THROTTLED

    def test_half_open_remain(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.HALF_OPEN)
        d = b.update(_snap(gpu=0.4, api=0.3, db=0.3, queue=0.2, error=2.0))
        assert d.mode == BreakerMode.HALF_OPEN

    def test_throttled_remain(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.THROTTLED)
        d = b.update(_snap(gpu=0.6, api=0.5, db=0.5, queue=0.3, error=3.0))
        assert d.mode == BreakerMode.THROTTLED

    def test_safe_stop_latched(self):
        b = BreakerStateMachine(_thresholds())
        b.update(_snap(gpu=0.99, api=0.99, db=0.99, queue=0.99, error=10.0))
        assert b.mode == BreakerMode.SAFE_STOP
        d = b.update(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.SAFE_STOP

    def test_safe_stop_from_any_mode(self):
        for start in [BreakerMode.CLOSED, BreakerMode.THROTTLED, BreakerMode.HALF_OPEN]:
            b = BreakerStateMachine(_thresholds())
            b.force_mode(start)
            d = b.update(_snap(gpu=0.99, api=0.99, db=0.99, queue=0.99, error=10.0))
            assert d.mode == BreakerMode.SAFE_STOP

    def test_reset_from_safe_stop(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.SAFE_STOP)
        d = b.reset(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.CLOSED

    def test_reset_denied_high_score(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.SAFE_STOP)
        d = b.reset(_snap(gpu=0.9, api=0.8, db=0.7, queue=0.5, error=5.0))
        assert d.mode == BreakerMode.SAFE_STOP
        assert "denied" in d.reason

    def test_overload_score_bounds(self):
        b = BreakerStateMachine(_thresholds())
        lo = b.overload_score(_snap(gpu=0, api=0, db=0, queue=0, error=0))
        hi = b.overload_score(_snap(gpu=1, api=1, db=1, queue=1, error=10))
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_open_mode_passthrough(self):
        b = BreakerStateMachine(_thresholds())
        b.force_mode(BreakerMode.OPEN)
        d = b.update(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.OPEN

    def test_thread_safety(self):
        b = BreakerStateMachine(_thresholds())
        errors = []
        def worker(i):
            try:
                for _ in range(100):
                    b.update(_snap(gpu=0.5 + 0.3 * (i % 2), api=0.3, db=0.3, queue=0.2, error=1.0))
            except Exception as e:
                errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors


# ═══════════════════════════════════════════════════════════════════════════
#  2. TRANSITION MODEL
# ═══════════════════════════════════════════════════════════════════════════

class TestTransitionModel:

    def test_closed_has_highest_caps(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets=_budgets())
        assert env.max_gpu_seconds > 0
        assert env.max_prompt_k == 64.0

    def test_mode_caps_monotonically_decrease(self):
        model = _test_model()
        modes = [BreakerMode.CLOSED, BreakerMode.THROTTLED, BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP]
        gpu_caps = [
            model.synthesize_envelope(snapshot=_snap(), mode=m, budgets=_budgets()).max_gpu_seconds
            for m in modes
        ]
        for i in range(len(gpu_caps) - 1):
            assert gpu_caps[i] >= gpu_caps[i + 1]

    def test_open_mode_zero_caps(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.OPEN, budgets=_budgets())
        assert env.max_gpu_seconds == 0.0
        assert env.max_external_api_calls == 0.0
        assert env.max_prompt_k == 0.0

    def test_saturated_state_collapses_caps(self):
        model = _test_model()
        snap = _snap(gpu=0.92, api=0.95, db=0.95, queue=0.95, error=10.0)
        env = model.synthesize_envelope(snapshot=snap, mode=BreakerMode.CLOSED, budgets=_budgets())
        assert env.max_gpu_seconds == 0.0
        assert env.max_external_api_calls == 0.0

    def test_budget_keys_canonical(self):
        model = _test_model()
        budgets = {"gpu_shift": 123.0, "external_api_shift": 45.0, "mutation_shift": 10.0}
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets=budgets)
        assert env.remaining_gpu_shift == 123.0
        assert env.remaining_external_api_shift == 45.0
        assert env.remaining_mutation_shift == 10.0

    def test_missing_budget_keys_default_zero(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets={})
        assert env.remaining_gpu_shift == 0.0

    def test_next_state_safe_pass(self):
        model = _test_model()
        receipt = ExecutionReceipt(action_id="a1", state_version=42,
                                   payload_digest="d", executed=True, observed_at_ns=time_ns())
        after = _snap(gpu=0.5, api=0.5, db=0.5, queue=0.5, error=5.0)
        assert model.next_state_safe(before=_snap(), emitted_action={}, receipt=receipt, after=after)

    def test_next_state_safe_fail_high_util(self):
        model = _test_model()
        receipt = ExecutionReceipt(action_id="a1", state_version=42,
                                   payload_digest="d", executed=True, observed_at_ns=time_ns())
        after = _snap(gpu=0.96)
        assert not model.next_state_safe(before=_snap(), emitted_action={}, receipt=receipt, after=after)

    def test_next_state_safe_fail_not_executed(self):
        model = _test_model()
        receipt = ExecutionReceipt(action_id="a1", state_version=42,
                                   payload_digest="d", executed=False, observed_at_ns=time_ns())
        assert not model.next_state_safe(before=_snap(), emitted_action={}, receipt=receipt, after=_snap())


# ═══════════════════════════════════════════════════════════════════════════
#  3. POLICY BUILDER → V5 INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestPolicyBuilderV5:

    @pytest.mark.parametrize("mode", [
        BreakerMode.CLOSED, BreakerMode.THROTTLED,
        BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP,
    ])
    def test_lint_parse_build(self, mode):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=mode, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        assert lint_config(config) == []
        PolicyParser().parse(config)
        system = NumerailSystem.from_config(config)
        assert system.region.n_dim == 30

    @pytest.mark.parametrize("mode", [
        BreakerMode.CLOSED, BreakerMode.THROTTLED,
        BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP,
    ])
    def test_enforce_and_guarantee(self, mode):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=mode, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        system = NumerailSystem.from_config(config)
        request = _safe_stop_request() if mode == BreakerMode.SAFE_STOP else _small_request()
        action = request.as_action_dict()
        action.update({k: 0.0 for k in env.trusted_context()})
        result = system.enforce(action, trusted_context=env.trusted_context())
        out = result.output
        if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            assert system.region.is_feasible(out.enforced_vector, tol=1e-6)

    def test_open_mode_rejects_nonzero_workload(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.OPEN, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        system = NumerailSystem.from_config(config)
        action = _small_request().as_action_dict()
        action.update({k: 0.0 for k in env.trusted_context()})
        result = system.enforce(action, trusted_context=env.trusted_context())
        assert result.output.result == EnforcementResult.REJECT

    def test_saturated_degenerate_feasible(self):
        model = _test_model()
        snap = _snap(gpu=0.95, api=0.95, db=0.95, queue=0.95, error=10.0)
        for mode in [BreakerMode.CLOSED, BreakerMode.THROTTLED, BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP]:
            env = model.synthesize_envelope(snapshot=snap, mode=mode, budgets=_budgets())
            config = build_v5_policy_from_envelope(env)
            system = NumerailSystem.from_config(config)
            feasible, _ = system.check_feasibility()
            assert feasible, f"Degenerate region infeasible for {mode.value}"

    def test_structural_external_le_internal(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        system = NumerailSystem.from_config(config)
        action = WorkloadRequest(
            prompt_k=5, completion_k=2, internal_tool_calls=2,
            external_api_calls=10, cloud_mutation_calls=1, gpu_seconds=10,
            parallel_workers=2, traffic_shift_pct=2, worker_scale_up_pct=2,
            feature_flag_changes=0, rollback_batch_pct=0,
            pager_notifications=1, customer_comms_count=0,
        ).as_action_dict()
        action.update({k: 0.0 for k in env.trusted_context()})
        result = system.enforce(action, trusted_context=env.trusted_context())
        out = result.output
        if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            vals = result.enforced_values
            assert vals["external_api_calls"] <= vals["internal_tool_calls"] + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
#  4. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:

    def _grant(self):
        return ExecutableGrant(
            action_id="a1", state_version=42,
            expires_at_ns=time_ns() + 5_000_000_000,
            reservation_token="tok_1", enforced_values={"x": 1.0},
            payload_digest="abc123",
        )

    def _receipt(self, **kw):
        base = dict(action_id="a1", state_version=42, payload_digest="abc123",
                    executed=True, observed_at_ns=time_ns())
        base.update(kw)
        return ExecutionReceipt(**base)

    def test_valid(self):
        validate_receipt_against_grant(grant=self._grant(), receipt=self._receipt())

    def test_action_id_mismatch(self):
        with pytest.raises(ReceiptValidationError, match="action_id"):
            validate_receipt_against_grant(grant=self._grant(), receipt=self._receipt(action_id="wrong"))

    def test_state_version_mismatch(self):
        with pytest.raises(ReceiptValidationError, match="state_version"):
            validate_receipt_against_grant(grant=self._grant(), receipt=self._receipt(state_version=99))

    def test_digest_mismatch(self):
        with pytest.raises(ReceiptValidationError, match="payload_digest"):
            validate_receipt_against_grant(grant=self._grant(), receipt=self._receipt(payload_digest="wrong"))

    def test_expired(self):
        grant = ExecutableGrant(action_id="a1", state_version=42, expires_at_ns=1000,
                                reservation_token="t", enforced_values={}, payload_digest="abc123")
        with pytest.raises(ReceiptValidationError, match="expires"):
            validate_receipt_against_grant(grant=grant, receipt=self._receipt(observed_at_ns=2000))


# ═══════════════════════════════════════════════════════════════════════════
#  5. LOCAL BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class TestLocalBackend:

    def test_no_config_raises(self):
        with pytest.raises(RuntimeError):
            LocalNumerailBackend().enforce(policy_id="p", proposed_action={},
                                          action_id="a", trusted_context=None, execution_topic=None)

    def test_budget_remaining_no_config(self):
        assert LocalNumerailBackend().budget_remaining() == {}

    def test_enforce_and_budget(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        backend = LocalNumerailBackend()
        backend.set_active_config(config)
        action = _small_request().as_action_dict()
        action.update({k: 0.0 for k in env.trusted_context()})
        result = backend.enforce(policy_id=env.policy_id, proposed_action=action,
                                 action_id="a1", trusted_context=env.trusted_context(),
                                 execution_topic=None)
        assert result["decision"] in ("approve", "project", "reject")
        assert "gpu_shift" in backend.budget_remaining()


# ═══════════════════════════════════════════════════════════════════════════
#  6. GOVERNOR LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════

class TestGovernor:

    def _gov(self):
        res = _MockReservationMgr()
        gov = StateTransitionGovernor(
            backend=LocalNumerailBackend(),
            transition_model=_test_model(),
            reservation_mgr=res, digestor=_MockDigestor(),
            thresholds=_thresholds(),
            bootstrap_budgets=_budgets(),
        )
        return gov, res

    def test_enforce_approve(self):
        gov, res = self._gov()
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")
        assert step.breaker.mode == BreakerMode.CLOSED
        assert step.numerail_result["decision"] in ("approve", "project")
        assert step.grant is not None
        assert len(res.acquired) == 1

    def test_first_cycle_uses_bootstrap_budgets(self):
        """Regression: first cycle must use bootstrap_budgets, not empty backend."""
        gov, _ = self._gov()
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="first")
        assert step.envelope.remaining_gpu_shift == 500.0
        assert step.numerail_result["decision"] in ("approve", "project")

    def test_no_bootstrap_no_backend_raises(self):
        """Regression: missing budget state must fail loud, not compile zero-budget policy."""
        gov = StateTransitionGovernor(
            backend=LocalNumerailBackend(),
            transition_model=_test_model(),
            reservation_mgr=_MockReservationMgr(), digestor=_MockDigestor(),
            thresholds=_thresholds(),
            # no bootstrap_budgets
        )
        with pytest.raises(RuntimeError, match="No budget state available"):
            gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")

    def test_enforce_reject_releases(self):
        gov, res = self._gov()
        big = WorkloadRequest(
            prompt_k=100, completion_k=50, internal_tool_calls=100,
            external_api_calls=50, cloud_mutation_calls=20, gpu_seconds=200,
            parallel_workers=20, traffic_shift_pct=80, worker_scale_up_pct=90,
            feature_flag_changes=30, rollback_batch_pct=60,
            pager_notifications=15, customer_comms_count=10,
        )
        step = gov.enforce_next_step(request=big, snapshot=_snap(), action_id="a1")
        assert step.numerail_result["decision"] == "reject"
        assert step.grant is None
        assert len(res.released) == 1

    def test_open_fast_path(self):
        gov, res = self._gov()
        gov.breaker.force_mode(BreakerMode.OPEN)
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")
        assert step.breaker.mode == BreakerMode.OPEN
        assert step.numerail_result["decision"] == "reject"
        assert step.grant is None
        assert len(res.acquired) == 0

    def test_stale_forces_safe_stop(self):
        gov, _ = self._gov()
        step = gov.enforce_next_step(request=_safe_stop_request(),
                                      snapshot=_snap(observed_at_ns=1), action_id="a1")
        assert step.breaker.mode == BreakerMode.SAFE_STOP

    def test_commit_success(self):
        gov, res = self._gov()
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")
        assert step.grant is not None
        receipt = ExecutionReceipt(
            action_id="a1", state_version=step.grant.state_version,
            payload_digest=step.grant.payload_digest,
            executed=True, observed_at_ns=time_ns(),
        )
        assert gov.commit(step=step, receipt=receipt, next_snapshot=_snap())
        assert len(res.committed) == 1

    def test_commit_unsafe_triggers_safe_stop(self):
        gov, res = self._gov()
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")
        assert step.grant is not None
        receipt = ExecutionReceipt(
            action_id="a1", state_version=step.grant.state_version,
            payload_digest=step.grant.payload_digest,
            executed=True, observed_at_ns=time_ns(),
        )
        assert not gov.commit(step=step, receipt=receipt, next_snapshot=_snap(gpu=0.96))
        assert gov.breaker.mode == BreakerMode.SAFE_STOP
        assert len(res.released) >= 1

    def test_commit_bad_receipt_raises(self):
        gov, res = self._gov()
        step = gov.enforce_next_step(request=_small_request(), snapshot=_snap(), action_id="a1")
        assert step.grant is not None
        bad = ExecutionReceipt(action_id="WRONG", state_version=step.grant.state_version,
                               payload_digest=step.grant.payload_digest,
                               executed=True, observed_at_ns=time_ns())
        with pytest.raises(ReceiptValidationError):
            gov.commit(step=step, receipt=bad, next_snapshot=_snap())
        assert len(res.released) >= 1

    def test_commit_no_grant(self):
        gov, _ = self._gov()
        step = GovernedStep(
            breaker=BreakerDecision(BreakerMode.OPEN, 0.0, "open"),
            envelope=_test_model().synthesize_envelope(
                snapshot=_snap(), mode=BreakerMode.OPEN, budgets=_budgets()),
            numerail_result={"decision": "reject"}, grant=None,
        )
        receipt = ExecutionReceipt(action_id="a1", state_version=42,
                                   payload_digest="d", executed=True, observed_at_ns=time_ns())
        assert not gov.commit(step=step, receipt=receipt, next_snapshot=_snap())

    def test_manual_reset(self):
        gov, _ = self._gov()
        gov.breaker.force_mode(BreakerMode.SAFE_STOP)
        d = gov.manual_reset(_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0))
        assert d.mode == BreakerMode.CLOSED

    def test_mode_transition_cycle(self):
        gov, _ = self._gov()
        assert gov.breaker.mode == BreakerMode.CLOSED
        gov.enforce_next_step(request=_small_request(),
                              snapshot=_snap(gpu=0.9, api=0.8, db=0.7, queue=0.5, error=5.0),
                              action_id="a1")
        assert gov.breaker.mode == BreakerMode.THROTTLED
        gov.enforce_next_step(request=_small_request(),
                              snapshot=_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0),
                              action_id="a2")
        assert gov.breaker.mode == BreakerMode.HALF_OPEN
        gov.enforce_next_step(request=_small_request(),
                              snapshot=_snap(gpu=0.1, api=0.1, db=0.1, queue=0.1, error=0.0),
                              action_id="a3")
        assert gov.breaker.mode == BreakerMode.CLOSED

    def test_policy_rebuilt_each_cycle(self):
        gov, _ = self._gov()
        s1 = gov.enforce_next_step(request=_small_request(), snapshot=_snap(gpu=0.45), action_id="a1")
        s2 = gov.enforce_next_step(request=_small_request(), snapshot=_snap(gpu=0.80), action_id="a2")
        assert s2.envelope.max_gpu_seconds < s1.envelope.max_gpu_seconds


# ═══════════════════════════════════════════════════════════════════════════
#  7. GUARANTEE FUZZING
# ═══════════════════════════════════════════════════════════════════════════

class TestGuaranteeFuzz:

    @pytest.mark.parametrize("seed", range(20))
    def test_random_workload(self, seed):
        rng = np.random.RandomState(seed)
        model = _test_model()
        snap = _snap(
            gpu=rng.uniform(0.1, 0.7), api=rng.uniform(0.1, 0.7),
            db=rng.uniform(0.1, 0.7), queue=rng.uniform(0.1, 0.7),
            error=rng.uniform(0.0, 5.0),
        )
        # Use Python-native random to select enum members — np.random.choice
        # returns numpy.str_ which loses enum identity.
        import random
        random.seed(seed)
        mode = random.choice([BreakerMode.CLOSED, BreakerMode.THROTTLED,
                              BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP])
        env = model.synthesize_envelope(snapshot=snap, mode=mode, budgets=_budgets())
        config = build_v5_policy_from_envelope(env)
        system = NumerailSystem.from_config(config)
        request = WorkloadRequest(
            prompt_k=rng.uniform(0, 30), completion_k=rng.uniform(0, 10),
            internal_tool_calls=rng.uniform(0, 20), external_api_calls=rng.uniform(0, 10),
            cloud_mutation_calls=rng.uniform(0, 5), gpu_seconds=rng.uniform(0, 60),
            parallel_workers=rng.uniform(0, 8), traffic_shift_pct=rng.uniform(0, 30),
            worker_scale_up_pct=rng.uniform(0, 30), feature_flag_changes=rng.uniform(0, 10),
            rollback_batch_pct=rng.uniform(0, 20), pager_notifications=rng.uniform(0, 5),
            customer_comms_count=rng.uniform(0, 3),
        )
        action = request.as_action_dict()
        action.update({k: 0.0 for k in env.trusted_context()})
        result = system.enforce(action, trusted_context=env.trusted_context())
        out = result.output
        if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            assert system.region.is_feasible(out.enforced_vector, tol=1e-6), \
                f"GUARANTEE VIOLATION: seed={seed}, mode={mode.value}"


# ═══════════════════════════════════════════════════════════════════════════
#  8. TYPES
# ═══════════════════════════════════════════════════════════════════════════

class TestTypes:

    def test_thresholds_valid(self):
        BreakerThresholds(trip_score=0.5, reset_score=0.3, safe_stop_score=0.8)

    def test_thresholds_invalid(self):
        with pytest.raises(ValueError):
            BreakerThresholds(trip_score=0.3, reset_score=0.5, safe_stop_score=0.8)

    def test_workload_as_action_dict(self):
        d = _small_request().as_action_dict()
        assert len(d) == 13
        assert all(isinstance(v, float) for v in d.values())

    def test_snapshot_frozen(self):
        with pytest.raises(AttributeError):
            _snap().current_gpu_util = 0.99

    def test_envelope_trusted_context_shape(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.CLOSED, budgets=_budgets())
        ctx = env.trusted_context()
        assert len(ctx) == 17
        assert ctx["current_gpu_util"] == 0.45


# ═══════════════════════════════════════════════════════════════════════════
#  9. MODE BOUNDARY NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class TestModeBoundary:

    def test_string_mode_accepted(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode="closed", budgets=_budgets())
        assert env.mode == BreakerMode.CLOSED
        assert "closed" in env.policy_id

    def test_all_string_modes(self):
        model = _test_model()
        for s in ("closed", "throttled", "half_open", "safe_stop", "open"):
            env = model.synthesize_envelope(snapshot=_snap(), mode=s, budgets=_budgets())
            assert env.mode == BreakerMode(s)

    def test_enum_mode_accepted(self):
        model = _test_model()
        env = model.synthesize_envelope(snapshot=_snap(), mode=BreakerMode.THROTTLED, budgets=_budgets())
        assert env.mode == BreakerMode.THROTTLED

    def test_invalid_string_raises(self):
        model = _test_model()
        with pytest.raises(ValueError):
            model.synthesize_envelope(snapshot=_snap(), mode="turbo", budgets=_budgets())

    def test_invalid_type_raises(self):
        model = _test_model()
        with pytest.raises(TypeError, match="must be BreakerMode or str"):
            model.synthesize_envelope(snapshot=_snap(), mode=42, budgets=_budgets())

    def test_numpy_str_mode_accepted(self):
        """Regression: np.random.choice on enum list returns numpy.str_."""
        model = _test_model()
        numpy_str = np.str_("closed")
        env = model.synthesize_envelope(snapshot=_snap(), mode=numpy_str, budgets=_budgets())
        assert env.mode == BreakerMode.CLOSED
