from __future__ import annotations

from dataclasses import dataclass
from time import time_ns
from typing import Mapping, Union

from .types import (
    BreakerMode,
    ExecutionReceipt,
    TelemetrySnapshot,
    TransitionEnvelope,
    TransitionModel,
)


# ── mode-dependent static caps (module-level constants) ──────────────────

_MODE_CAPS: dict[BreakerMode, dict[str, float]] = {
    BreakerMode.CLOSED: dict(
        prompt_k=64.0, completion_k=16.0, internal_tool_calls=40.0,
        external_api_calls=20.0, cloud_mutation_calls=10.0, gpu_seconds=120.0,
        parallel_workers=16.0, traffic_shift_pct=60.0, worker_scale_up_pct=80.0,
        feature_flag_changes=20.0, rollback_batch_pct=50.0,
        pager_notifications=10.0, customer_comms_count=5.0,
    ),
    BreakerMode.THROTTLED: dict(
        prompt_k=40.0, completion_k=12.0, internal_tool_calls=18.0,
        external_api_calls=10.0, cloud_mutation_calls=4.0, gpu_seconds=60.0,
        parallel_workers=8.0, traffic_shift_pct=25.0, worker_scale_up_pct=30.0,
        feature_flag_changes=4.0, rollback_batch_pct=20.0,
        pager_notifications=6.0, customer_comms_count=2.0,
    ),
    BreakerMode.HALF_OPEN: dict(
        prompt_k=20.0, completion_k=6.0, internal_tool_calls=8.0,
        external_api_calls=4.0, cloud_mutation_calls=2.0, gpu_seconds=20.0,
        parallel_workers=3.0, traffic_shift_pct=10.0, worker_scale_up_pct=10.0,
        feature_flag_changes=1.0, rollback_batch_pct=10.0,
        pager_notifications=4.0, customer_comms_count=1.0,
    ),
    BreakerMode.SAFE_STOP: dict(
        prompt_k=8.0, completion_k=2.0, internal_tool_calls=4.0,
        external_api_calls=3.0, cloud_mutation_calls=1.0, gpu_seconds=10.0,
        parallel_workers=1.0, traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
        feature_flag_changes=0.0, rollback_batch_pct=10.0,
        pager_notifications=3.0, customer_comms_count=1.0,
    ),
}

_WORKLOAD_FIELDS: tuple[str, ...] = (
    "prompt_k", "completion_k", "internal_tool_calls",
    "external_api_calls", "cloud_mutation_calls", "gpu_seconds",
    "parallel_workers", "traffic_shift_pct", "worker_scale_up_pct",
    "feature_flag_changes", "rollback_batch_pct",
    "pager_notifications", "customer_comms_count",
)


def _normalize_mode(mode: Union[BreakerMode, str]) -> BreakerMode:
    """Canonicalize mode to a BreakerMode enum member.

    Accepts BreakerMode instances, plain strings, and numpy string scalars.
    Raises TypeError for anything else.
    """
    if isinstance(mode, BreakerMode):
        return mode
    if isinstance(mode, str):
        return BreakerMode(str(mode))
    raise TypeError(f"mode must be BreakerMode or str, got {type(mode).__name__}")


@dataclass(frozen=True)
class IncidentCommanderTransitionModel(TransitionModel):
    """Conservative one-step model for an autonomous incident commander.

    The model is intentionally simple and outer-bounding.  Its purpose is not
    to solve general reachability, but to compile a trusted, conservative
    envelope into a V5-compatible convex policy.

    Budget keys consumed here use the **canonical V5 names** produced by the
    policy builder: ``gpu_shift``, ``external_api_shift``, ``mutation_shift``.
    """

    freshness_ns: int = 5_000_000_000  # 5 seconds

    # ── envelope synthesis ───────────────────────────────────────────

    def synthesize_envelope(
        self,
        *,
        snapshot: TelemetrySnapshot,
        mode: Union[BreakerMode, str],
        budgets: Mapping[str, float],
    ) -> TransitionEnvelope:
        mode = _normalize_mode(mode)
        now = time_ns()
        expires_at = now + self.freshness_ns

        caps = dict(_MODE_CAPS.get(mode, {f: 0.0 for f in _WORKLOAD_FIELDS}))

        # State-derived physical limits
        gpu_max = max(
            0.0,
            300.0 * (0.88 - snapshot.current_gpu_util)
            - snapshot.ctrl_gpu_reserve_seconds
            - snapshot.gpu_disturbance_margin_seconds,
        )
        api_max = max(
            0.0,
            50.0 * (0.90 - snapshot.current_api_util)
            - snapshot.ctrl_api_reserve_calls
            - snapshot.api_disturbance_margin_calls,
        )
        parallel_max = max(0.0, 24.0 - snapshot.ctrl_parallel_reserve)
        mutation_max = max(0.0, 10.0 - snapshot.ctrl_cloud_mutation_reserve)

        db_headroom = (
            0.92 - snapshot.current_db_util
            - snapshot.db_disturbance_margin_pct / 100.0
        )
        traffic_max = max(0.0, db_headroom / 0.004)
        scale_max = max(0.0, db_headroom / 0.003)
        rollback_max = max(0.0, db_headroom / 0.002)

        queue_headroom = (
            0.90 - snapshot.current_queue_util
            - snapshot.queue_disturbance_margin_pct / 100.0
        )
        queue_parallel_max = max(0.0, queue_headroom / 0.004)

        return TransitionEnvelope(
            policy_id=f"incident-cb::{mode.value}",
            mode=mode,
            max_prompt_k=min(caps["prompt_k"], 64.0),
            max_completion_k=min(caps["completion_k"], 16.0),
            max_internal_tool_calls=min(caps["internal_tool_calls"], 40.0),
            max_external_api_calls=min(caps["external_api_calls"], api_max),
            max_cloud_mutation_calls=min(caps["cloud_mutation_calls"], mutation_max),
            max_gpu_seconds=min(caps["gpu_seconds"], gpu_max),
            max_parallel_workers=min(caps["parallel_workers"], parallel_max, queue_parallel_max),
            max_traffic_shift_pct=min(caps["traffic_shift_pct"], traffic_max),
            max_worker_scale_up_pct=min(caps["worker_scale_up_pct"], scale_max),
            max_feature_flag_changes=min(caps["feature_flag_changes"], 20.0),
            max_rollback_batch_pct=min(caps["rollback_batch_pct"], rollback_max),
            max_pager_notifications=min(caps["pager_notifications"], 10.0),
            max_customer_comms_count=min(caps["customer_comms_count"], 5.0),
            # Canonical V5 budget keys — no fallback aliases
            remaining_gpu_shift=float(budgets.get("gpu_shift", 0.0)),
            remaining_external_api_shift=float(budgets.get("external_api_shift", 0.0)),
            remaining_mutation_shift=float(budgets.get("mutation_shift", 0.0)),
            min_required_state_version=int(snapshot.state_version),
            expires_at_ns=int(expires_at),
            trusted=snapshot,
        )

    # ── post-execution safety check ──────────────────────────────────

    def next_state_safe(
        self,
        *,
        before: TelemetrySnapshot,
        emitted_action: Mapping[str, float],
        receipt: ExecutionReceipt,
        after: TelemetrySnapshot,
    ) -> bool:
        if not receipt.executed:
            return False
        return (
            after.current_gpu_util <= 0.95
            and after.current_api_util <= 0.95
            and after.current_db_util <= 0.95
            and after.current_queue_util <= 0.95
            and after.current_error_rate_pct <= 15.0
        )
