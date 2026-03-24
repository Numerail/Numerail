from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol


class BreakerMode(str, Enum):
    """Operational mode for supervisory degradation."""

    CLOSED = "closed"
    THROTTLED = "throttled"
    HALF_OPEN = "half_open"
    SAFE_STOP = "safe_stop"
    OPEN = "open"


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Trusted current state :math:`s_t` used to synthesize a one-step envelope.

    All fields are runtime facts or protected reserves. None are model-controlled.
    """

    state_version: int
    observed_at_ns: int

    current_gpu_util: float
    current_api_util: float
    current_db_util: float
    current_queue_util: float
    current_error_rate_pct: float

    ctrl_gpu_reserve_seconds: float
    ctrl_api_reserve_calls: float
    ctrl_parallel_reserve: float
    ctrl_cloud_mutation_reserve: float

    gpu_disturbance_margin_seconds: float
    api_disturbance_margin_calls: float
    db_disturbance_margin_pct: float
    queue_disturbance_margin_pct: float


@dataclass(frozen=True)
class WorkloadRequest:
    """Model-proposed workload authority for the next step."""

    prompt_k: float
    completion_k: float
    internal_tool_calls: float
    external_api_calls: float
    cloud_mutation_calls: float
    gpu_seconds: float
    parallel_workers: float
    traffic_shift_pct: float
    worker_scale_up_pct: float
    feature_flag_changes: float
    rollback_batch_pct: float
    pager_notifications: float
    customer_comms_count: float

    def as_action_dict(self) -> dict[str, float]:
        return {
            "prompt_k": float(self.prompt_k),
            "completion_k": float(self.completion_k),
            "internal_tool_calls": float(self.internal_tool_calls),
            "external_api_calls": float(self.external_api_calls),
            "cloud_mutation_calls": float(self.cloud_mutation_calls),
            "gpu_seconds": float(self.gpu_seconds),
            "parallel_workers": float(self.parallel_workers),
            "traffic_shift_pct": float(self.traffic_shift_pct),
            "worker_scale_up_pct": float(self.worker_scale_up_pct),
            "feature_flag_changes": float(self.feature_flag_changes),
            "rollback_batch_pct": float(self.rollback_batch_pct),
            "pager_notifications": float(self.pager_notifications),
            "customer_comms_count": float(self.customer_comms_count),
        }


@dataclass(frozen=True)
class BreakerThresholds:
    """Hysteresis thresholds for breaker-state transitions."""

    trip_score: float
    reset_score: float
    safe_stop_score: float

    def __post_init__(self) -> None:
        if not (self.reset_score <= self.trip_score <= self.safe_stop_score):
            raise ValueError(
                "BreakerThresholds must satisfy reset_score <= trip_score <= safe_stop_score"
            )


@dataclass(frozen=True)
class BreakerDecision:
    """Mode decision produced from trusted telemetry."""

    mode: BreakerMode
    overload_score: float
    reason: str


@dataclass(frozen=True)
class TransitionEnvelope:
    """Conservative one-step admissible envelope compiled from trusted state.

    The envelope is deliberately V5-friendly: it contains only mode-dependent
    ceilings, time/freshness preconditions, current budget ceilings, and the
    trusted state needed to build a convex policy.
    """

    policy_id: str
    mode: BreakerMode

    max_prompt_k: float
    max_completion_k: float
    max_internal_tool_calls: float
    max_external_api_calls: float
    max_cloud_mutation_calls: float
    max_gpu_seconds: float
    max_parallel_workers: float
    max_traffic_shift_pct: float
    max_worker_scale_up_pct: float
    max_feature_flag_changes: float
    max_rollback_batch_pct: float
    max_pager_notifications: float
    max_customer_comms_count: float

    remaining_gpu_shift: float
    remaining_external_api_shift: float
    remaining_mutation_shift: float

    min_required_state_version: int
    expires_at_ns: int

    trusted: TelemetrySnapshot

    def trusted_context(self) -> dict[str, float]:
        return {
            "state_version": float(self.trusted.state_version),
            "observed_at_ns": float(self.trusted.observed_at_ns),
            "min_required_state_version": float(self.min_required_state_version),
            "expires_at_ns": float(self.expires_at_ns),
            "current_gpu_util": float(self.trusted.current_gpu_util),
            "current_api_util": float(self.trusted.current_api_util),
            "current_db_util": float(self.trusted.current_db_util),
            "current_queue_util": float(self.trusted.current_queue_util),
            "current_error_rate_pct": float(self.trusted.current_error_rate_pct),
            "ctrl_gpu_reserve_seconds": float(self.trusted.ctrl_gpu_reserve_seconds),
            "ctrl_api_reserve_calls": float(self.trusted.ctrl_api_reserve_calls),
            "ctrl_parallel_reserve": float(self.trusted.ctrl_parallel_reserve),
            "ctrl_cloud_mutation_reserve": float(self.trusted.ctrl_cloud_mutation_reserve),
            "gpu_disturbance_margin_seconds": float(self.trusted.gpu_disturbance_margin_seconds),
            "api_disturbance_margin_calls": float(self.trusted.api_disturbance_margin_calls),
            "db_disturbance_margin_pct": float(self.trusted.db_disturbance_margin_pct),
            "queue_disturbance_margin_pct": float(self.trusted.queue_disturbance_margin_pct),
        }


@dataclass(frozen=True)
class ExecutableGrant:
    """Runtime-binding artifact for an emitted Numerail action."""

    action_id: str
    state_version: int
    expires_at_ns: int
    reservation_token: str
    enforced_values: Mapping[str, float]
    payload_digest: str


@dataclass(frozen=True)
class ExecutionReceipt:
    """Attested execution result returned by the runtime layer."""

    action_id: str
    state_version: int
    payload_digest: str
    executed: bool
    observed_at_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernedStep:
    """Complete outcome of one supervisory enforcement attempt."""

    breaker: BreakerDecision
    envelope: TransitionEnvelope
    numerail_result: Mapping[str, Any]
    grant: Optional[ExecutableGrant]


class TransitionModel(Protocol):
    """Trusted, conservative state-transition analysis interface."""

    def synthesize_envelope(
        self,
        *,
        snapshot: TelemetrySnapshot,
        mode: BreakerMode,
        budgets: Mapping[str, float],
    ) -> TransitionEnvelope:
        ...

    def next_state_safe(
        self,
        *,
        before: TelemetrySnapshot,
        emitted_action: Mapping[str, float],
        receipt: ExecutionReceipt,
        after: TelemetrySnapshot,
    ) -> bool:
        ...


class ReservationManager(Protocol):
    """Interface for atomicity / freshness protection around enforcement."""

    def acquire(
        self,
        *,
        state_version: int,
        expires_at_ns: int,
        resource_claims: Mapping[str, float],
    ) -> str:
        ...

    def commit(self, *, token: str, receipt: ExecutionReceipt) -> None:
        ...

    def release(self, *, token: str) -> None:
        ...


class Digestor(Protocol):
    def digest(self, payload: Mapping[str, Any]) -> str:
        ...


class NumerailBackend(Protocol):
    """Minimal backend seam so the governor can support local and service paths.

    Budget keys returned by budget_remaining() must match the V5 BudgetSpec.name
    values produced by the policy builder: ``gpu_shift``, ``external_api_shift``,
    ``mutation_shift``.
    """

    def budget_remaining(self) -> Mapping[str, float]:
        ...

    def set_active_config(self, config: Mapping[str, Any]) -> None:
        ...

    def enforce(
        self,
        *,
        policy_id: str,
        proposed_action: Mapping[str, float],
        action_id: str,
        trusted_context: Optional[Mapping[str, float]],
        execution_topic: Optional[str],
    ) -> Mapping[str, Any]:
        ...

    def rollback(self, *, action_id: str) -> Any:
        ...
