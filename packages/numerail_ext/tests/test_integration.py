"""Integration tests: full governor lifecycle exercising both numerail and numerail_ext.

These tests exercise the complete path a real deployment would run:
  build_global_default() → PolicyParser → NumerailSystem → BreakerStateMachine
  → StateTransitionGovernor → 20+ enforcement cycles with escalating workload
  → breaker trip → envelope tightening → budget depletion → rollback
  → recovery → audit verification → contract round-trip

Each test function covers one aspect of the lifecycle in isolation, using shared
fixture helpers and the LocalNumerailBackend (in-memory, no external dependencies).
"""

from __future__ import annotations

import hashlib
import json
from time import time_ns
from typing import Any, Mapping

import numpy as np
import pytest

# core
from numerail.engine import EnforcementResult, NumerailSystem
from numerail.parser import PolicyParser

# ext
from numerail_ext.survivability.breaker import BreakerStateMachine
from numerail_ext.survivability.contract import NumerailPolicyContract
from numerail_ext.survivability.global_default import build_global_default
from numerail_ext.survivability.governor import StateTransitionGovernor
from numerail_ext.survivability.local_backend import LocalNumerailBackend
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
# 120 seconds — allows snapshot re-use across steps in a single test function
_FRESHNESS_NS = 120_000_000_000


# ── Shared helpers ────────────────────────────────────────────────────────────

def _snap(
    gpu: float = 0.30,
    api: float = 0.30,
    db: float = 0.35,
    queue: float = 0.20,
    error: float = 0.5,
    state_version: int = 1,
) -> TelemetrySnapshot:
    """Create a telemetry snapshot. All utilisation values in [0,1]; error in %."""
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
    """A small, structurally-valid workload request (satisfies all relation constraints)."""
    return WorkloadRequest(
        prompt_k=5.0 * scale,
        completion_k=2.0 * scale,
        internal_tool_calls=5.0 * scale,   # >= external_api_calls and >= cloud_mutation_calls
        external_api_calls=3.0 * scale,
        cloud_mutation_calls=1.0 * scale,
        gpu_seconds=10.0 * scale,
        parallel_workers=2.0 * scale,
        traffic_shift_pct=0.0,
        worker_scale_up_pct=0.0,
        feature_flag_changes=0.0,
        rollback_batch_pct=0.0,
        pager_notifications=1.0,           # pager + comms <= external_api_calls
        customer_comms_count=0.0,
    )


class _MockReservationMgr:
    """In-memory reservation manager — acquires tokens, commits, and releases them."""

    def __init__(self):
        self._counter = 0

    def acquire(self, *, state_version, expires_at_ns, resource_claims) -> str:
        self._counter += 1
        return f"tok_{self._counter}"

    def commit(self, *, token, receipt) -> None:
        pass

    def release(self, *, token) -> None:
        pass


class _MockDigestor:
    """SHA-256 digestor used by the governor to produce payload digests."""

    def digest(self, payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


def _make_governor(bootstrap_budgets: dict | None = None) -> StateTransitionGovernor:
    """Construct a governor backed by in-memory infrastructure."""
    return StateTransitionGovernor(
        backend=LocalNumerailBackend(),
        transition_model=IncidentCommanderTransitionModel(freshness_ns=_FRESHNESS_NS),
        reservation_mgr=_MockReservationMgr(),
        digestor=_MockDigestor(),
        thresholds=_THRESHOLDS,
        bootstrap_budgets=bootstrap_budgets or dict(_BOOTSTRAP),
    )


def _receipt(grant, snap: TelemetrySnapshot) -> ExecutionReceipt:
    """Build a synthetic execution receipt for a grant."""
    return ExecutionReceipt(
        action_id=grant.action_id,
        state_version=grant.state_version,
        payload_digest=grant.payload_digest,
        executed=True,
        observed_at_ns=snap.observed_at_ns,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_policy_parse_and_system_construction():
    """build_global_default() → PolicyParser().parse() → NumerailSystem.from_config().

    Verifies the three-stage pipeline produces a functional NumerailSystem and
    that enforcement returns a valid outcome for a zero-value proposal.
    """
    config = build_global_default()
    parsed = PolicyParser().parse(config)
    sys = NumerailSystem.from_config(parsed)

    assert len(sys._schema.fields) > 0, "Schema must have at least one field"

    # A zero-value proposal is structurally valid; enforcement must not raise
    zero_proposal = {f: 0.0 for f in sys._schema.fields}
    result = sys.enforce(zero_proposal)
    assert result.output.result in (
        EnforcementResult.APPROVE,
        EnforcementResult.PROJECT,
        EnforcementResult.REJECT,
    )


def test_breaker_initial_state_closed():
    """Both the governor's internal breaker and a bare BreakerStateMachine start in CLOSED."""
    gov = _make_governor()
    assert gov.breaker.mode == BreakerMode.CLOSED

    bare = BreakerStateMachine(_THRESHOLDS)
    assert bare.mode == BreakerMode.CLOSED


def test_breaker_trips_to_throttled_on_workload_escalation():
    """Over 25 enforcement cycles with GPU utilisation rising linearly from 0.30 to 0.95
    (and elevated api/db/queue/error), the breaker must trip to THROTTLED.

    Overload score weights: 0.30*gpu + 0.25*api + 0.20*db + 0.10*queue + 0.15*(error/100).
    With api=0.70, db=0.70, queue=0.50, error=5.0 the base contribution is ~0.373;
    the trip threshold (0.60) is crossed at gpu ≈ 0.76, which occurs around step 17.
    """
    gov = _make_governor()
    modes_seen: set[BreakerMode] = set()

    for step_idx in range(25):
        gpu_util = 0.30 + step_idx * (0.65 / 24)   # linearly 0.30 → 0.95
        snap = _snap(gpu=gpu_util, api=0.70, db=0.70, queue=0.50, error=5.0)
        step = gov.enforce_next_step(
            request=_request(),
            snapshot=snap,
            action_id=f"esc_{step_idx}",
        )
        modes_seen.add(step.breaker.mode)
        if step.breaker.mode == BreakerMode.THROTTLED:
            break  # trip confirmed; no need to continue escalation

    assert BreakerMode.THROTTLED in modes_seen, (
        f"Breaker did not trip to THROTTLED during GPU escalation. "
        f"Modes observed: {modes_seen}"
    )


def test_throttled_envelope_caps_tighter_than_closed():
    """Every workload field cap produced by THROTTLED mode must be ≤ the
    corresponding cap in CLOSED mode (for the same telemetry snapshot and budgets).

    This verifies the monotone authority-reduction property of the transition model.
    """
    model = IncidentCommanderTransitionModel(freshness_ns=_FRESHNESS_NS)
    snap = _snap()
    budgets = dict(_BOOTSTRAP)

    env_closed = model.synthesize_envelope(snapshot=snap, mode=BreakerMode.CLOSED, budgets=budgets)
    env_throttled = model.synthesize_envelope(snapshot=snap, mode=BreakerMode.THROTTLED, budgets=budgets)

    cap_pairs = [
        ("max_prompt_k",            env_closed.max_prompt_k,            env_throttled.max_prompt_k),
        ("max_completion_k",        env_closed.max_completion_k,        env_throttled.max_completion_k),
        ("max_internal_tool_calls", env_closed.max_internal_tool_calls, env_throttled.max_internal_tool_calls),
        ("max_external_api_calls",  env_closed.max_external_api_calls,  env_throttled.max_external_api_calls),
        ("max_cloud_mutation_calls",env_closed.max_cloud_mutation_calls,env_throttled.max_cloud_mutation_calls),
        ("max_gpu_seconds",         env_closed.max_gpu_seconds,         env_throttled.max_gpu_seconds),
        ("max_parallel_workers",    env_closed.max_parallel_workers,    env_throttled.max_parallel_workers),
        ("max_traffic_shift_pct",   env_closed.max_traffic_shift_pct,   env_throttled.max_traffic_shift_pct),
        ("max_worker_scale_up_pct", env_closed.max_worker_scale_up_pct, env_throttled.max_worker_scale_up_pct),
        ("max_feature_flag_changes",env_closed.max_feature_flag_changes,env_throttled.max_feature_flag_changes),
        ("max_rollback_batch_pct",  env_closed.max_rollback_batch_pct,  env_throttled.max_rollback_batch_pct),
        ("max_pager_notifications", env_closed.max_pager_notifications, env_throttled.max_pager_notifications),
        ("max_customer_comms_count",env_closed.max_customer_comms_count,env_throttled.max_customer_comms_count),
    ]

    for field, closed_cap, throttled_cap in cap_pairs:
        assert throttled_cap <= closed_cap + 1e-9, (
            f"THROTTLED cap for {field} ({throttled_cap:.4f}) exceeds "
            f"CLOSED cap ({closed_cap:.4f})"
        )


def test_enforcement_guarantee_holds_over_20_cycles():
    """For every APPROVE/PROJECT output across 22 enforcement cycles in CLOSED mode,
    the enforced vector must satisfy region.is_feasible() — the live verification of
    Theorem 1 against the full 30-field governor policy.

    For each step the exact envelope config used for enforcement is rebuilt into a fresh
    NumerailSystem; is_feasible() is called on the enforced vector using that system's
    region, which contains every constraint the engine evaluated.
    """
    gov = _make_governor()
    approved_or_projected = 0

    for i in range(22):
        snap = _snap(gpu=0.30)   # healthy utilisation → stays in CLOSED
        step = gov.enforce_next_step(
            request=_request(),
            snapshot=snap,
            action_id=f"g_{i}",
        )

        decision = step.numerail_result["decision"]
        if decision in {"approve", "project"}:
            approved_or_projected += 1
            enforced = step.numerail_result["enforced_values"]

            # Rebuild NumerailSystem from the exact envelope config used this step,
            # then call is_feasible on the enforced vector.
            config = build_v5_policy_from_envelope(step.envelope)
            sys_check = NumerailSystem.from_config(config)
            fields = sys_check._schema.fields
            vec = np.array([enforced[f] for f in fields], dtype=float)
            region = sys_check._versions.current

            assert region.is_feasible(vec), (
                f"Guarantee violated at step {i}: decision={decision}, "
                f"enforced vector is not in the feasible region"
            )

    assert approved_or_projected >= 18, (
        f"Only {approved_or_projected}/22 steps were approved/projected. "
        f"Policy may be over-restrictive or test setup is incorrect."
    )


def test_budget_depletion_restricts_output():
    """After a small GPU budget is drained over two enforcement cycles, a large
    GPU request is capped at or below the remaining budget allowance.

    The budget constraint (remaining_gpu_shift: gpu_seconds ≤ X) is encoded as a
    linear row in the envelope policy and limits the enforced output regardless of
    the original proposal size.
    """
    small_bootstrap = {"gpu_shift": 25.0, "external_api_shift": 100.0, "mutation_shift": 40.0}
    gov = _make_governor(bootstrap_budgets=small_bootstrap)
    snap = _snap()

    # Drain the GPU budget over two cycles (~10 GPU-s per cycle)
    for i in range(2):
        drain_step = gov.enforce_next_step(
            request=_request(),
            snapshot=snap,
            action_id=f"drain_{i}",
        )
        if drain_step.grant is not None:
            gov.commit(
                step=drain_step,
                receipt=_receipt(drain_step.grant, snap),
                next_snapshot=snap,
            )

    # Read the remaining budget that the NEXT enforcement cycle will start with
    remaining_for_big = gov.backend.budget_remaining().get("gpu_shift", 0.0)

    # Propose 50 GPU-s — far beyond any remaining allowance
    big_request = WorkloadRequest(
        prompt_k=5.0, completion_k=2.0, internal_tool_calls=5.0,
        external_api_calls=3.0, cloud_mutation_calls=1.0,
        gpu_seconds=50.0,
        parallel_workers=2.0, traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
        feature_flag_changes=0.0, rollback_batch_pct=0.0,
        pager_notifications=1.0, customer_comms_count=0.0,
    )
    big_step = gov.enforce_next_step(
        request=big_request,
        snapshot=snap,
        action_id="big_gpu",
    )

    if big_step.grant is not None:
        enforced_gpu = big_step.grant.enforced_values["gpu_seconds"]
        assert enforced_gpu <= remaining_for_big + 1e-6, (
            f"Budget not enforced: enforced_gpu={enforced_gpu:.4f} "
            f"exceeds remaining={remaining_for_big:.4f}"
        )
    # grant=None (REJECT) is also correct: it means the engine could not project
    # the proposal to a point that satisfies all constraints simultaneously


def test_rollback_restores_budget_exactly():
    """Rolling back an action restores the GPU budget by exactly the enforced
    consumption delta — neither more nor less."""
    gov = _make_governor()
    snap = _snap()

    step = gov.enforce_next_step(request=_request(), snapshot=snap, action_id="rb_target")
    assert step.grant is not None, (
        "Expected APPROVE/PROJECT for a small request on healthy infrastructure"
    )

    enforced_gpu = step.grant.enforced_values["gpu_seconds"]
    budget_after_enforce = gov.backend.budget_remaining()["gpu_shift"]

    gov.backend.rollback(action_id="rb_target")
    budget_after_rollback = gov.backend.budget_remaining()["gpu_shift"]

    delta_restored = budget_after_rollback - budget_after_enforce
    assert abs(delta_restored - enforced_gpu) < 1e-6, (
        f"Rollback delta mismatch: restored={delta_restored:.6f}, "
        f"expected enforced GPU-s={enforced_gpu:.6f}"
    )


def test_recovery_toward_closed_on_low_utilization():
    """After tripping to THROTTLED, dropping utilisation well below the reset
    threshold drives the breaker through HALF_OPEN and back to CLOSED.

    The overload score for the low-stress snapshot is ≈ 0.085, well below
    reset_score=0.30, so the mode should recover in two consecutive updates.
    """
    gov = _make_governor()

    # Trip: score ≈ 0.68 (above trip_score=0.60, below safe_stop_score=0.85)
    # 0.30*0.80 + 0.25*0.70 + 0.20*0.70 + 0.10*0.50 + 0.15*(5.0/10.0) = 0.68
    high_snap = _snap(gpu=0.80, api=0.70, db=0.70, queue=0.50, error=5.0)
    trip_step = gov.enforce_next_step(
        request=_request(), snapshot=high_snap, action_id="trip"
    )
    assert trip_step.breaker.mode == BreakerMode.THROTTLED, (
        f"Expected THROTTLED after high-stress snapshot, got {trip_step.breaker.mode}"
    )

    # Recovery: score ≈ 0.085 (well below reset_score=0.30)
    low_snap = _snap(gpu=0.10, api=0.10, db=0.10, queue=0.10, error=0.1)

    # First low-stress update: THROTTLED → HALF_OPEN
    step2 = gov.enforce_next_step(
        request=_request(), snapshot=low_snap, action_id="recover_1"
    )
    assert step2.breaker.mode in (BreakerMode.HALF_OPEN, BreakerMode.CLOSED), (
        f"Expected HALF_OPEN or CLOSED after first low-stress update, "
        f"got {step2.breaker.mode}"
    )

    # Second low-stress update: HALF_OPEN → CLOSED
    step3 = gov.enforce_next_step(
        request=_request(), snapshot=low_snap, action_id="recover_2"
    )
    assert step3.breaker.mode == BreakerMode.CLOSED, (
        f"Expected CLOSED after sustained low utilisation, got {step3.breaker.mode}"
    )


def test_audit_chain_integrity():
    """The audit chain on a NumerailSystem accumulates tamper-evident hash-linked
    records; verify_audit() must confirm integrity after multiple enforcement cycles.

    Uses NumerailSystem.from_config() directly (not the governor's LocalNumerailBackend)
    because NumerailSystem exposes the verify_audit() / AuditChain API.
    """
    config = build_global_default()
    parsed = PolicyParser().parse(config)
    sys = NumerailSystem.from_config(parsed)

    n_cycles = 6
    zero_proposal = {f: 0.0 for f in sys._schema.fields}
    for i in range(n_cycles):
        sys.enforce(zero_proposal, action_id=f"audit_{i}")

    ok, depth = sys.verify_audit()
    assert ok, "Audit chain verification failed — records may have been tampered with"
    assert depth == n_cycles, (
        f"Expected {n_cycles} audit records, got {depth}"
    )


def test_contract_round_trip_digest_stability():
    """A NumerailPolicyContract serialised to a dict and reconstructed with from_dict()
    must reproduce an identical content digest — no field is lost or mutated.

    This verifies both the SHA-256 content-addressing and the canonical serialisation
    round-trip of the policy contract.
    """
    config = build_global_default()
    contract = NumerailPolicyContract.from_v5_config(
        config,
        author_id="integration-test",
        policy_id="global-default::integration-v1.0",
    )

    assert contract.verify_digest(), "Original contract digest is invalid"

    d = contract.to_dict()
    contract2 = NumerailPolicyContract.from_dict(d)

    assert contract2.verify_digest(), "Round-tripped contract digest is invalid"
    assert contract.policy_id == contract2.policy_id, "policy_id changed across round-trip"
    assert d["digest"] == contract2.to_dict()["digest"], (
        "Payload digest changed across round-trip — canonical serialisation is unstable"
    )
