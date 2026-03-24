"""Build an exact Numerail V5-compatible config from a synthesized envelope.

The generated policy contains:
  - 30 schema fields (13 workload + 4 sequencing + 5 state + 4 reserve + 4 margin)
  - ~78 linear constraints (box bounds, structural, reserve-aware headroom, budgets)
  - 1 quadratic constraint (workload energy bound)
  - 1 SOCP constraint (joint infrastructure burst envelope)
  - 1 PSD constraint (coupled cross-channel headroom)
  - 17 trusted fields (all non-workload dimensions)
  - Dimension policies: FORBIDDEN on trusted/sequencing, FLAG on sensitive workload
  - 3 shift budgets with nonnegative consumption

Freshness architecture (three layers — documented here for auditors):
  - V5 freshness constraints enforce *internal envelope consistency*:
    ``state_version >= min_required_state_version`` and
    ``observed_at_ns <= expires_at_ns``.  These catch corrupted or replayed
    trusted-context payloads.
  - The governor's ``time_ns()`` check enforces *real-clock freshness*.
    It catches stale telemetry before enforcement begins.
  - ``ReservationManager.acquire(expires_at_ns)`` enforces *execution-time
    freshness*.  It catches enforcement cycles that take too long.
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import TransitionEnvelope


def build_v5_policy_from_envelope(env: TransitionEnvelope) -> dict[str, Any]:
    """Build an exact Numerail V5-compatible config from a synthesized envelope."""

    fields = [
        # workload
        "prompt_k",
        "completion_k",
        "internal_tool_calls",
        "external_api_calls",
        "cloud_mutation_calls",
        "gpu_seconds",
        "parallel_workers",
        "traffic_shift_pct",
        "worker_scale_up_pct",
        "feature_flag_changes",
        "rollback_batch_pct",
        "pager_notifications",
        "customer_comms_count",
        # sequencing / freshness
        "state_version",
        "observed_at_ns",
        "min_required_state_version",
        "expires_at_ns",
        # trusted live state
        "current_gpu_util",
        "current_api_util",
        "current_db_util",
        "current_queue_util",
        "current_error_rate_pct",
        # protected control-plane reserve
        "ctrl_gpu_reserve_seconds",
        "ctrl_api_reserve_calls",
        "ctrl_parallel_reserve",
        "ctrl_cloud_mutation_reserve",
        # uncertainty margins
        "gpu_disturbance_margin_seconds",
        "api_disturbance_margin_calls",
        "db_disturbance_margin_pct",
        "queue_disturbance_margin_pct",
    ]
    idx = {f: i for i, f in enumerate(fields)}
    n = len(fields)

    def row(coeffs: Mapping[str, float]) -> list[float]:
        r = [0.0] * n
        for k, v in coeffs.items():
            r[idx[k]] = float(v)
        return r

    A: list[list[float]] = []
    b: list[float] = []
    names: list[str] = []

    def add_le(name: str, coeffs: Mapping[str, float], rhs: float) -> None:
        A.append(row(coeffs))
        b.append(float(rhs))
        names.append(name)

    def add_box(field: str, lo: float, hi: float) -> None:
        add_le(f"max_{field}", {field: 1.0}, hi)
        add_le(f"min_{field}", {field: -1.0}, -lo)

    # Workload bounds derived from the envelope
    add_box("prompt_k", 0.0, env.max_prompt_k)
    add_box("completion_k", 0.0, env.max_completion_k)
    add_box("internal_tool_calls", 0.0, env.max_internal_tool_calls)
    add_box("external_api_calls", 0.0, env.max_external_api_calls)
    add_box("cloud_mutation_calls", 0.0, env.max_cloud_mutation_calls)
    add_box("gpu_seconds", 0.0, env.max_gpu_seconds)
    add_box("parallel_workers", 0.0, env.max_parallel_workers)
    add_box("traffic_shift_pct", 0.0, env.max_traffic_shift_pct)
    add_box("worker_scale_up_pct", 0.0, env.max_worker_scale_up_pct)
    add_box("feature_flag_changes", 0.0, env.max_feature_flag_changes)
    add_box("rollback_batch_pct", 0.0, env.max_rollback_batch_pct)
    add_box("pager_notifications", 0.0, env.max_pager_notifications)
    add_box("customer_comms_count", 0.0, env.max_customer_comms_count)

    # Trusted and sequencing fields get realistic nonnegative bounds.
    for field, hi in (
        ("state_version", 1e21),
        ("observed_at_ns", 1e21),
        ("min_required_state_version", 1e21),
        ("expires_at_ns", 1e21),
        ("current_gpu_util", 1.0),
        ("current_api_util", 1.0),
        ("current_db_util", 1.0),
        ("current_queue_util", 1.0),
        ("current_error_rate_pct", 100.0),
        ("ctrl_gpu_reserve_seconds", 1e6),
        ("ctrl_api_reserve_calls", 1e6),
        ("ctrl_parallel_reserve", 1e6),
        ("ctrl_cloud_mutation_reserve", 1e6),
        ("gpu_disturbance_margin_seconds", 1e6),
        ("api_disturbance_margin_calls", 1e6),
        ("db_disturbance_margin_pct", 100.0),
        ("queue_disturbance_margin_pct", 100.0),
    ):
        add_box(field, 0.0, hi)

    # Freshness / sequencing conditions (internal envelope consistency)
    add_le(
        "state_version_fresh_enough",
        {"min_required_state_version": 1.0, "state_version": -1.0},
        0.0,
    )
    add_le(
        "observation_not_expired",
        {"observed_at_ns": 1.0, "expires_at_ns": -1.0},
        0.0,
    )

    # Structural relations
    add_le("external_le_internal_tools", {"external_api_calls": 1.0, "internal_tool_calls": -1.0}, 0.0)
    add_le("mutations_le_internal_tools", {"cloud_mutation_calls": 1.0, "internal_tool_calls": -1.0}, 0.0)
    add_le("flag_changes_require_mutations", {"feature_flag_changes": 1.0, "cloud_mutation_calls": -5.0}, 0.0)
    add_le("rollback_requires_mutations", {"rollback_batch_pct": 1.0, "cloud_mutation_calls": -25.0}, 0.0)
    add_le("traffic_shift_requires_mutations", {"traffic_shift_pct": 1.0, "cloud_mutation_calls": -20.0}, 0.0)
    add_le(
        "customer_and_pager_within_external_api",
        {"pager_notifications": 1.0, "customer_comms_count": 1.0, "external_api_calls": -1.0},
        0.0,
    )
    add_le("customer_impact_step_total", {"traffic_shift_pct": 1.0, "rollback_batch_pct": 1.0}, 100.0)

    # Reserve-aware headroom
    add_le(
        "gpu_headroom_with_reserve",
        {
            "current_gpu_util": 1.0,
            "gpu_seconds": 1.0 / 300.0,
            "ctrl_gpu_reserve_seconds": 1.0 / 300.0,
            "gpu_disturbance_margin_seconds": 1.0 / 300.0,
        },
        0.88,
    )
    add_le(
        "api_headroom_with_reserve",
        {
            "current_api_util": 1.0,
            "external_api_calls": 1.0 / 50.0,
            "ctrl_api_reserve_calls": 1.0 / 50.0,
            "api_disturbance_margin_calls": 1.0 / 50.0,
        },
        0.90,
    )
    add_le(
        "db_headroom_with_margin",
        {
            "current_db_util": 1.0,
            "traffic_shift_pct": 0.004,
            "worker_scale_up_pct": 0.003,
            "rollback_batch_pct": 0.002,
            "db_disturbance_margin_pct": 1.0 / 100.0,
        },
        0.92,
    )
    add_le(
        "queue_headroom_with_margin",
        {
            "current_queue_util": 1.0,
            "parallel_workers": 0.004,
            "queue_disturbance_margin_pct": 1.0 / 100.0,
        },
        0.90,
    )
    add_le("parallel_with_reserve", {"parallel_workers": 1.0, "ctrl_parallel_reserve": 1.0}, 24.0)
    add_le(
        "mutations_with_reserve",
        {"cloud_mutation_calls": 1.0, "ctrl_cloud_mutation_reserve": 1.0},
        10.0,
    )

    # Dynamic budget rows
    add_le("remaining_gpu_shift", {"gpu_seconds": 1.0}, env.remaining_gpu_shift)
    add_le("remaining_external_api_shift", {"external_api_calls": 1.0}, env.remaining_external_api_shift)
    add_le("remaining_mutation_shift", {"cloud_mutation_calls": 1.0}, env.remaining_mutation_shift)

    # Quadratic: workload energy bound
    Q = [[0.0] * n for _ in range(n)]
    for field, denom in {
        "prompt_k": 64.0,
        "completion_k": 16.0,
        "internal_tool_calls": 40.0,
        "external_api_calls": 20.0,
        "cloud_mutation_calls": 10.0,
        "gpu_seconds": 120.0,
        "parallel_workers": 16.0,
        "traffic_shift_pct": 60.0,
        "worker_scale_up_pct": 80.0,
        "feature_flag_changes": 20.0,
        "rollback_batch_pct": 50.0,
        "pager_notifications": 10.0,
        "customer_comms_count": 5.0,
    }.items():
        Q[idx[field]][idx[field]] = 1.0 / (denom * denom)

    quadratic_constraints = [{
        "Q": Q,
        "a": [0.0] * n,
        "b": 4.0,
        "name": "workload_energy",
    }]

    # SOCP: joint burst envelope
    M = [[0.0] * n for _ in range(6)]
    M[0][idx["current_gpu_util"]] = 1.0
    M[0][idx["gpu_seconds"]] = 1.0 / 300.0
    M[0][idx["ctrl_gpu_reserve_seconds"]] = 1.0 / 300.0
    M[0][idx["gpu_disturbance_margin_seconds"]] = 1.0 / 300.0

    M[1][idx["current_api_util"]] = 1.0
    M[1][idx["external_api_calls"]] = 1.0 / 50.0
    M[1][idx["ctrl_api_reserve_calls"]] = 1.0 / 50.0
    M[1][idx["api_disturbance_margin_calls"]] = 1.0 / 50.0

    M[2][idx["current_db_util"]] = 1.0
    M[2][idx["traffic_shift_pct"]] = 0.004
    M[2][idx["worker_scale_up_pct"]] = 0.003
    M[2][idx["rollback_batch_pct"]] = 0.002
    M[2][idx["db_disturbance_margin_pct"]] = 1.0 / 100.0

    M[3][idx["current_queue_util"]] = 1.0
    M[3][idx["parallel_workers"]] = 0.004
    M[3][idx["queue_disturbance_margin_pct"]] = 1.0 / 100.0

    M[4][idx["parallel_workers"]] = 1.0 / 24.0
    M[4][idx["ctrl_parallel_reserve"]] = 1.0 / 24.0

    M[5][idx["cloud_mutation_calls"]] = 1.0 / 10.0
    M[5][idx["ctrl_cloud_mutation_reserve"]] = 1.0 / 10.0

    socp_constraints = [{
        "M": M,
        "q": [0.0] * 6,
        "c": [0.0] * n,
        "d": 1.75,
        "name": "infrastructure_burst_envelope",
    }]

    # PSD: coupled headroom
    def zero33() -> list[list[float]]:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    A0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    A_list = [zero33() for _ in fields]

    # 1 - gpu path
    A_list[idx["current_gpu_util"]] = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["gpu_seconds"]] = [[-1.0 / 300.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["ctrl_gpu_reserve_seconds"]] = [[-1.0 / 300.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["gpu_disturbance_margin_seconds"]] = [[-1.0 / 300.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # 1 - api path
    A_list[idx["current_api_util"]] = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["external_api_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / 50.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["ctrl_api_reserve_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / 50.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["api_disturbance_margin_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / 50.0, 0.0], [0.0, 0.0, 0.0]]

    # 1 - queue path and couplings
    A_list[idx["current_queue_util"]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    A_list[idx["parallel_workers"]] = [[0.0, 0.003, 0.0], [0.003, 0.0, 0.0], [0.0, 0.0, -0.004]]
    A_list[idx["ctrl_parallel_reserve"]] = [[0.0, 0.003, 0.0], [0.003, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["traffic_shift_pct"]] = [[0.0, 0.003, 0.002], [0.003, 0.0, 0.0], [0.002, 0.0, 0.0]]
    A_list[idx["worker_scale_up_pct"]] = [[0.0, 0.003, 0.0], [0.003, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["rollback_batch_pct"]] = [[0.002, 0.0, 0.002], [0.0, 0.0, 0.0], [0.002, 0.0, 0.0]]
    A_list[idx["feature_flag_changes"]] = [[0.0, 0.0, 0.0], [0.0, 0.004, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["queue_disturbance_margin_pct"]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0 / 100.0]]

    psd_constraints = [{
        "A0": A0,
        "A_list": A_list,
        "name": "coupled_headroom_psd",
    }]

    return {
        "policy_id": env.policy_id,
        "schema": {"fields": fields},
        "polytope": {"A": A, "b": b, "names": names},
        "quadratic_constraints": quadratic_constraints,
        "socp_constraints": socp_constraints,
        "psd_constraints": psd_constraints,
        "trusted_fields": [
            "state_version",
            "observed_at_ns",
            "min_required_state_version",
            "expires_at_ns",
            "current_gpu_util",
            "current_api_util",
            "current_db_util",
            "current_queue_util",
            "current_error_rate_pct",
            "ctrl_gpu_reserve_seconds",
            "ctrl_api_reserve_calls",
            "ctrl_parallel_reserve",
            "ctrl_cloud_mutation_reserve",
            "gpu_disturbance_margin_seconds",
            "api_disturbance_margin_calls",
            "db_disturbance_margin_pct",
            "queue_disturbance_margin_pct",
        ],
        "enforcement": {
            "mode": "project",
            "dimension_policies": {
                "state_version": "forbidden",
                "observed_at_ns": "forbidden",
                "min_required_state_version": "forbidden",
                "expires_at_ns": "forbidden",
                "current_gpu_util": "forbidden",
                "current_api_util": "forbidden",
                "current_db_util": "forbidden",
                "current_queue_util": "forbidden",
                "current_error_rate_pct": "forbidden",
                "ctrl_gpu_reserve_seconds": "forbidden",
                "ctrl_api_reserve_calls": "forbidden",
                "ctrl_parallel_reserve": "forbidden",
                "ctrl_cloud_mutation_reserve": "forbidden",
                "gpu_disturbance_margin_seconds": "forbidden",
                "api_disturbance_margin_calls": "forbidden",
                "db_disturbance_margin_pct": "forbidden",
                "queue_disturbance_margin_pct": "forbidden",
                "parallel_workers": "project_with_flag",
                "traffic_shift_pct": "project_with_flag",
                "worker_scale_up_pct": "project_with_flag",
                "feature_flag_changes": "project_with_flag",
                "rollback_batch_pct": "project_with_flag",
                "cloud_mutation_calls": "project_with_flag",
            },
            "routing_thresholds": {
                "silent": 0.05,
                "flagged": 1.5,
                "confirmation": 4.0,
                "hard_reject": 8.0,
            },
        },
        "budgets": [
            {
                "name": "gpu_shift",
                "constraint_name": "remaining_gpu_shift",
                "weight": {"gpu_seconds": 1.0},
                "initial": env.remaining_gpu_shift,
                "consumption_mode": "nonnegative",
            },
            {
                "name": "external_api_shift",
                "constraint_name": "remaining_external_api_shift",
                "weight": {"external_api_calls": 1.0},
                "initial": env.remaining_external_api_shift,
                "consumption_mode": "nonnegative",
            },
            {
                "name": "mutation_shift",
                "constraint_name": "remaining_mutation_shift",
                "weight": {"cloud_mutation_calls": 1.0},
                "initial": env.remaining_mutation_shift,
                "consumption_mode": "nonnegative",
            },
        ],
    }
