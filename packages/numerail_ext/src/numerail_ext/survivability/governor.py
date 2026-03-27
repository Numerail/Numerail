"""V5-aligned supervisory governor for time-aware safe execution.

Responsibilities sit *around* the unchanged V5 kernel:
  1. Choose breaker mode from trusted telemetry.
  2. Synthesize a conservative one-step envelope.
  3. Generate a V5-compatible policy from that envelope.
  4. Enforce via unchanged V5 interfaces.
  5. Bind exact actuation to the emitted action.
  6. Validate the realized next state.

Freshness architecture (three layers):
  - V5 constraints enforce internal envelope consistency
    (state_version >= min_required, observed_at <= expires_at).
    Catches corrupted or replayed payloads.
  - Governor ``time_ns()`` check enforces real-clock freshness.
    Catches stale telemetry.
  - ``ReservationManager.acquire(expires_at_ns)`` enforces execution-time
    freshness.  Catches slow enforcement cycles.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from time import time_ns
from typing import Any, Mapping, Optional

from .breaker import BreakerStateMachine
from .policy_builder import build_v5_policy_from_envelope
from .types import (
    BreakerMode,
    BreakerThresholds,
    Digestor,
    ExecutionReceipt,
    ExecutableGrant,
    GovernedStep,
    NumerailBackend,
    ReservationManager,
    TelemetrySnapshot,
    TransitionModel,
    WorkloadRequest,
)
from .validation import validate_receipt_against_grant


@dataclass
class StateTransitionGovernor:
    """Supervisory enforcement governor.

    The governor never modifies the V5 kernel.  It only controls *which policy*
    the kernel enforces against.  V5's post-check remains the trust boundary.

    ``bootstrap_budgets`` provides the initial budget state for the first
    enforcement cycle, before the backend has loaded any config.  If omitted
    and no live backend state exists, the governor raises RuntimeError rather
    than silently compiling a zero-budget policy.
    """

    backend: NumerailBackend
    transition_model: TransitionModel
    reservation_mgr: ReservationManager
    digestor: Digestor
    thresholds: BreakerThresholds
    bootstrap_budgets: Optional[Mapping[str, float]] = None
    trusted_context_provider: Optional[Any] = None  # TrustedContextProvider

    def __post_init__(self) -> None:
        self.breaker = BreakerStateMachine(self.thresholds)

    def _resolve_budgets(self) -> Mapping[str, float]:
        """Resolve current budget state. Fail loud if uninitialized."""
        live = self.backend.budget_remaining()
        if live:
            return live
        if self.bootstrap_budgets:
            return self.bootstrap_budgets
        raise RuntimeError(
            "No budget state available. Provide bootstrap_budgets at governor "
            "construction or preload the backend with set_active_config()."
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _placeholder_action(request: WorkloadRequest) -> dict[str, float]:
        """Merge model-proposed workload with zero placeholders for trusted fields.

        The zeros are overwritten by ``trusted_context`` before V5 vectorizes.
        If a trusted field were ever missing from the policy's trusted_fields
        declaration, the zero would reach the constraint checker and the
        freshness constraints would cause REJECT — fail-safe by design.
        """
        action = request.as_action_dict()
        action.update(
            {
                "state_version": 0.0,
                "observed_at_ns": 0.0,
                "min_required_state_version": 0.0,
                "expires_at_ns": 0.0,
                "current_gpu_util": 0.0,
                "current_api_util": 0.0,
                "current_db_util": 0.0,
                "current_queue_util": 0.0,
                "current_error_rate_pct": 0.0,
                "ctrl_gpu_reserve_seconds": 0.0,
                "ctrl_api_reserve_calls": 0.0,
                "ctrl_parallel_reserve": 0.0,
                "ctrl_cloud_mutation_reserve": 0.0,
                "gpu_disturbance_margin_seconds": 0.0,
                "api_disturbance_margin_calls": 0.0,
                "db_disturbance_margin_pct": 0.0,
                "queue_disturbance_margin_pct": 0.0,
            }
        )
        return action

    @staticmethod
    def _resource_claims_from_envelope(envelope: Any) -> dict[str, float]:
        return {
            "gpu_seconds_cap": float(envelope.max_gpu_seconds),
            "external_api_cap": float(envelope.max_external_api_calls),
            "mutation_cap": float(envelope.max_cloud_mutation_calls),
            "parallel_cap": float(envelope.max_parallel_workers),
        }

    # ── primary API ──────────────────────────────────────────────────

    def enforce_next_step(
        self,
        *,
        request: WorkloadRequest,
        snapshot: TelemetrySnapshot,
        action_id: str,
        execution_topic: str = "runtime",
    ) -> GovernedStep:
        """Evaluate telemetry, synthesize envelope, enforce through V5."""

        # If a TrustedContextProvider is configured, overwrite snapshot fields
        # with server-authoritative values before any other processing.
        # The provider is the authority; the snapshot is the carrier.
        if self.trusted_context_provider is not None:
            tc = self.trusted_context_provider.get_trusted_context()
            replace_kwargs: dict[str, Any] = {}
            for field_name, value in tc.items():
                if hasattr(snapshot, field_name):
                    existing = getattr(snapshot, field_name)
                    replace_kwargs[field_name] = type(existing)(value)
            if replace_kwargs:
                snapshot = dataclasses.replace(snapshot, **replace_kwargs)

        # Real-clock freshness check.  Forces SAFE_STOP if telemetry is stale.
        # This is a supervisory precondition, not a V5 kernel theorem.
        freshness_ns = getattr(self.transition_model, "freshness_ns", None)
        if freshness_ns is not None and snapshot.observed_at_ns + int(freshness_ns) < time_ns():
            self.breaker.force_mode(BreakerMode.SAFE_STOP)
            breaker_decision = self.breaker.update(snapshot)
        else:
            breaker_decision = self.breaker.update(snapshot)

        envelope = self.transition_model.synthesize_envelope(
            snapshot=snapshot,
            mode=breaker_decision.mode,
            budgets=self._resolve_budgets(),
        )

        # OPEN mode fast-path: all authority suspended, skip V5 cycle entirely.
        # Returning REJECT preserves V5's guarantee vacuously — REJECT makes
        # no feasibility claim.
        if breaker_decision.mode == BreakerMode.OPEN:
            return GovernedStep(
                breaker=breaker_decision,
                envelope=envelope,
                numerail_result={
                    "decision": "reject",
                    "enforced_values": None,
                    "feedback": {"message": "breaker OPEN: all authority suspended"},
                    "audit_hash": None,
                    "action_id": action_id,
                },
                grant=None,
            )

        # Build and load a fresh policy every cycle.  The envelope ceilings
        # depend on current telemetry (not just breaker mode), so a cached
        # policy from the same mode but different utilization would be stale.
        config = build_v5_policy_from_envelope(envelope)
        self.backend.set_active_config(config)

        reservation_token = self.reservation_mgr.acquire(
            state_version=snapshot.state_version,
            expires_at_ns=envelope.expires_at_ns,
            resource_claims=self._resource_claims_from_envelope(envelope),
        )

        numerail_result = self.backend.enforce(
            policy_id=envelope.policy_id,
            proposed_action=self._placeholder_action(request),
            action_id=action_id,
            trusted_context=envelope.trusted_context(),
            execution_topic=execution_topic,
        )

        grant: Optional[ExecutableGrant] = None
        if numerail_result["decision"] in {"approve", "project"}:
            enforced_values = dict(numerail_result["enforced_values"])
            payload_digest = self.digestor.digest(
                {
                    "action_id": action_id,
                    "state_version": snapshot.state_version,
                    "reservation_token": reservation_token,
                    **enforced_values,
                }
            )
            grant = ExecutableGrant(
                action_id=action_id,
                state_version=snapshot.state_version,
                expires_at_ns=envelope.expires_at_ns,
                reservation_token=reservation_token,
                enforced_values=enforced_values,
                payload_digest=payload_digest,
            )
        else:
            self.reservation_mgr.release(token=reservation_token)

        return GovernedStep(
            breaker=breaker_decision,
            envelope=envelope,
            numerail_result=numerail_result,
            grant=grant,
        )

    # ── commit / rollback ────────────────────────────────────────────

    def commit(
        self,
        *,
        step: GovernedStep,
        receipt: ExecutionReceipt,
        next_snapshot: TelemetrySnapshot,
        rollback_on_failure: bool = True,
    ) -> bool:
        """Validate execution receipt and post-execution state.

        Returns True if the receipt is valid and the next state is safe.
        On failure: releases the reservation, optionally rolls back the
        V5 budget, and (for unsafe next-state) latches SAFE_STOP.
        """
        if step.grant is None:
            return False

        try:
            validate_receipt_against_grant(grant=step.grant, receipt=receipt)
        except Exception:
            self.reservation_mgr.release(token=step.grant.reservation_token)
            if rollback_on_failure:
                self.backend.rollback(action_id=step.grant.action_id)
            raise

        safe = self.transition_model.next_state_safe(
            before=step.envelope.trusted,
            emitted_action=step.grant.enforced_values,
            receipt=receipt,
            after=next_snapshot,
        )

        if not safe:
            self.reservation_mgr.release(token=step.grant.reservation_token)
            if rollback_on_failure:
                self.backend.rollback(action_id=step.grant.action_id)
            self.breaker.force_mode(BreakerMode.SAFE_STOP)
            return False

        self.reservation_mgr.commit(token=step.grant.reservation_token, receipt=receipt)
        return True

    def rollback(self, *, action_id: str) -> Any:
        """Roll back a previously enforced action's budget consumption."""
        return self.backend.rollback(action_id=action_id)

    # ── recovery ─────────────────────────────────────────────────────

    def manual_reset(self, snapshot: TelemetrySnapshot):
        """Manual recovery from SAFE_STOP.  Requires healthy telemetry.

        Delegates to ``BreakerStateMachine.reset()``, which only transitions
        to CLOSED if the overload score is at or below the reset threshold.
        """
        return self.breaker.reset(snapshot)
