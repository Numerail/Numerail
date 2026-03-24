"""Numerail Global Default Policy Pack v1.0

Conservative, locked-down-by-default policy configuration for AI actuation
governance.  Ships as the standard exemplar for Numerail V5 and the breaker
suite.  Designed to satisfy the technical requirements of the EU AI Act
Articles 9 (risk management), 12 (record-keeping), 14 (human oversight),
and 15 (accuracy, robustness, cybersecurity) for high-risk AI systems that
govern infrastructure resources.

Architecture:
    7 modules produce a single V5-compatible config dict with 30 schema
    fields, ~80 linear constraints, 1 quadratic constraint, 1 SOCP
    constraint, 1 PSD constraint, 17 trusted fields, dimension policies,
    routing thresholds, and 3 shift budgets.

Usage:
    Standalone V5:
        from numerail_ext.survivability.global_default import build_global_default
        config = build_global_default()
        system = NumerailSystem.from_config(config)

    With breaker suite (as bootstrap):
        config = build_global_default()
        governor = StateTransitionGovernor(
            ...,
            bootstrap_budgets={
                "gpu_shift": config["_meta"]["budget_defaults"]["gpu_shift"],
                "external_api_shift": config["_meta"]["budget_defaults"]["external_api_shift"],
                "mutation_shift": config["_meta"]["budget_defaults"]["mutation_shift"],
            },
        )

    Custom overrides:
        config = build_global_default(
            gpu_seconds_cap=60.0,        # tighter than default 120
            safety_margin=0.90,           # 10% buffer
            shift_gpu_budget=300.0,       # smaller shift budget
        )

Design principle:
    The default is denial.  Every permission must be explicitly granted
    through constraint geometry.  Deployers relax constraints to match
    their governance intent — they do not add constraints they might forget.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEMA: 30 fields partitioned into model-proposed and server-authoritative
# ═══════════════════════════════════════════════════════════════════════════

# 13 workload dimensions — the AI proposes these
WORKLOAD_FIELDS = (
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
)

# 4 sequencing / freshness fields — server-authoritative
SEQUENCING_FIELDS = (
    "state_version",
    "observed_at_ns",
    "min_required_state_version",
    "expires_at_ns",
)

# 5 live state fields — server-authoritative
STATE_FIELDS = (
    "current_gpu_util",
    "current_api_util",
    "current_db_util",
    "current_queue_util",
    "current_error_rate_pct",
)

# 4 protected reserve fields — server-authoritative
RESERVE_FIELDS = (
    "ctrl_gpu_reserve_seconds",
    "ctrl_api_reserve_calls",
    "ctrl_parallel_reserve",
    "ctrl_cloud_mutation_reserve",
)

# 4 disturbance margin fields — server-authoritative
MARGIN_FIELDS = (
    "gpu_disturbance_margin_seconds",
    "api_disturbance_margin_calls",
    "db_disturbance_margin_pct",
    "queue_disturbance_margin_pct",
)

ALL_FIELDS = WORKLOAD_FIELDS + SEQUENCING_FIELDS + STATE_FIELDS + RESERVE_FIELDS + MARGIN_FIELDS
TRUSTED_FIELDS = SEQUENCING_FIELDS + STATE_FIELDS + RESERVE_FIELDS + MARGIN_FIELDS

assert len(ALL_FIELDS) == 30
assert len(set(ALL_FIELDS)) == 30
assert len(TRUSTED_FIELDS) == 17


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT CAPS — conservative per-field ceilings
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_WORKLOAD_CAPS: dict[str, float] = {
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
}

# Trusted field upper bounds — realistic nonnegative ceilings
_TRUSTED_BOUNDS: dict[str, float] = {
    "state_version": 1e21,
    "observed_at_ns": 1e21,
    "min_required_state_version": 1e21,
    "expires_at_ns": 1e21,
    "current_gpu_util": 1.0,
    "current_api_util": 1.0,
    "current_db_util": 1.0,
    "current_queue_util": 1.0,
    "current_error_rate_pct": 100.0,
    "ctrl_gpu_reserve_seconds": 1e6,
    "ctrl_api_reserve_calls": 1e6,
    "ctrl_parallel_reserve": 1e6,
    "ctrl_cloud_mutation_reserve": 1e6,
    "gpu_disturbance_margin_seconds": 1e6,
    "api_disturbance_margin_calls": 1e6,
    "db_disturbance_margin_pct": 100.0,
    "queue_disturbance_margin_pct": 100.0,
}

# Default shift budgets
_DEFAULT_BUDGETS: dict[str, float] = {
    "gpu_shift": 3600.0,         # 1 hour of GPU seconds per shift
    "external_api_shift": 500.0,  # 500 external API calls per shift
    "mutation_shift": 100.0,      # 100 cloud mutations per shift
}

# Default reserve protection
_DEFAULT_RESERVES: dict[str, float] = {
    "ctrl_gpu_reserve_seconds": 30.0,
    "ctrl_api_reserve_calls": 5.0,
    "ctrl_parallel_reserve": 4.0,
    "ctrl_cloud_mutation_reserve": 2.0,
}

# Default disturbance margins
_DEFAULT_MARGINS: dict[str, float] = {
    "gpu_disturbance_margin_seconds": 15.0,
    "api_disturbance_margin_calls": 3.0,
    "db_disturbance_margin_pct": 5.0,
    "queue_disturbance_margin_pct": 3.0,
}

# Infrastructure capacity constants
_GPU_CAPACITY_SECONDS = 300.0
_GPU_HEADROOM_CEILING = 0.88
_API_CAPACITY_CALLS = 50.0
_API_HEADROOM_CEILING = 0.90
_DB_HEADROOM_CEILING = 0.92
_QUEUE_HEADROOM_CEILING = 0.90
_PARALLEL_CAPACITY = 24.0
_MUTATION_CAPACITY = 10.0

# Quadratic energy bound
_WORKLOAD_ENERGY_BOUND = 4.0

# SOCP burst bound
_BURST_NORM_BOUND = 1.75


# ═══════════════════════════════════════════════════════════════════════════
#  BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_global_default(
    *,
    # ── Module 1: Action authority overrides ──
    prompt_k_cap: float = _DEFAULT_WORKLOAD_CAPS["prompt_k"],
    completion_k_cap: float = _DEFAULT_WORKLOAD_CAPS["completion_k"],
    internal_tool_calls_cap: float = _DEFAULT_WORKLOAD_CAPS["internal_tool_calls"],
    external_api_calls_cap: float = _DEFAULT_WORKLOAD_CAPS["external_api_calls"],
    cloud_mutation_calls_cap: float = _DEFAULT_WORKLOAD_CAPS["cloud_mutation_calls"],
    gpu_seconds_cap: float = _DEFAULT_WORKLOAD_CAPS["gpu_seconds"],
    parallel_workers_cap: float = _DEFAULT_WORKLOAD_CAPS["parallel_workers"],
    traffic_shift_pct_cap: float = _DEFAULT_WORKLOAD_CAPS["traffic_shift_pct"],
    worker_scale_up_pct_cap: float = _DEFAULT_WORKLOAD_CAPS["worker_scale_up_pct"],
    feature_flag_changes_cap: float = _DEFAULT_WORKLOAD_CAPS["feature_flag_changes"],
    rollback_batch_pct_cap: float = _DEFAULT_WORKLOAD_CAPS["rollback_batch_pct"],
    pager_notifications_cap: float = _DEFAULT_WORKLOAD_CAPS["pager_notifications"],
    customer_comms_count_cap: float = _DEFAULT_WORKLOAD_CAPS["customer_comms_count"],
    # ── Module 5: Reserve and margin overrides ──
    gpu_reserve: float = _DEFAULT_RESERVES["ctrl_gpu_reserve_seconds"],
    api_reserve: float = _DEFAULT_RESERVES["ctrl_api_reserve_calls"],
    parallel_reserve: float = _DEFAULT_RESERVES["ctrl_parallel_reserve"],
    mutation_reserve: float = _DEFAULT_RESERVES["ctrl_cloud_mutation_reserve"],
    gpu_margin: float = _DEFAULT_MARGINS["gpu_disturbance_margin_seconds"],
    api_margin: float = _DEFAULT_MARGINS["api_disturbance_margin_calls"],
    db_margin: float = _DEFAULT_MARGINS["db_disturbance_margin_pct"],
    queue_margin: float = _DEFAULT_MARGINS["queue_disturbance_margin_pct"],
    # ── Module 7: Budget overrides ──
    shift_gpu_budget: float = _DEFAULT_BUDGETS["gpu_shift"],
    shift_api_budget: float = _DEFAULT_BUDGETS["external_api_shift"],
    shift_mutation_budget: float = _DEFAULT_BUDGETS["mutation_shift"],
    # ── Enforcement configuration ──
    safety_margin: float = 1.0,
    workload_energy_bound: float = _WORKLOAD_ENERGY_BOUND,
    burst_norm_bound: float = _BURST_NORM_BOUND,
    # ── Policy identity ──
    policy_id: str = "global-default::v1.0",
) -> dict[str, Any]:
    """Build the Global Default Policy Pack as a V5-compatible config dict.

    All parameters have conservative defaults.  Deployers override specific
    caps, reserves, margins, or budgets to match their governance intent.
    """

    fields = list(ALL_FIELDS)
    idx = {f: i for i, f in enumerate(fields)}
    n = len(fields)

    # ── Helpers ──────────────────────────────────────────────────────

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

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 1: Action Authority Bounds (EU AI Act Article 9)
    # ══════════════════════════════════════════════════════════════════

    caps = {
        "prompt_k": prompt_k_cap,
        "completion_k": completion_k_cap,
        "internal_tool_calls": internal_tool_calls_cap,
        "external_api_calls": external_api_calls_cap,
        "cloud_mutation_calls": cloud_mutation_calls_cap,
        "gpu_seconds": gpu_seconds_cap,
        "parallel_workers": parallel_workers_cap,
        "traffic_shift_pct": traffic_shift_pct_cap,
        "worker_scale_up_pct": worker_scale_up_pct_cap,
        "feature_flag_changes": feature_flag_changes_cap,
        "rollback_batch_pct": rollback_batch_pct_cap,
        "pager_notifications": pager_notifications_cap,
        "customer_comms_count": customer_comms_count_cap,
    }
    for field_name, hi in caps.items():
        add_box(field_name, 0.0, hi)

    # Trusted field bounds
    for field_name, hi in _TRUSTED_BOUNDS.items():
        add_box(field_name, 0.0, hi)

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 2: Structural Invariants (EU AI Act Article 9(4))
    # ══════════════════════════════════════════════════════════════════

    # Prerequisite relations — encode operational knowledge
    add_le(
        "external_le_internal_tools",
        {"external_api_calls": 1.0, "internal_tool_calls": -1.0}, 0.0,
    )
    add_le(
        "mutations_le_internal_tools",
        {"cloud_mutation_calls": 1.0, "internal_tool_calls": -1.0}, 0.0,
    )
    add_le(
        "flag_changes_require_mutations",
        {"feature_flag_changes": 1.0, "cloud_mutation_calls": -5.0}, 0.0,
    )
    add_le(
        "rollback_requires_mutations",
        {"rollback_batch_pct": 1.0, "cloud_mutation_calls": -25.0}, 0.0,
    )
    add_le(
        "traffic_shift_requires_mutations",
        {"traffic_shift_pct": 1.0, "cloud_mutation_calls": -20.0}, 0.0,
    )
    add_le(
        "customer_and_pager_within_external_api",
        {"pager_notifications": 1.0, "customer_comms_count": 1.0, "external_api_calls": -1.0}, 0.0,
    )
    add_le(
        "customer_impact_step_total",
        {"traffic_shift_pct": 1.0, "rollback_batch_pct": 1.0}, 100.0,
    )

    # Freshness / sequencing (internal envelope consistency)
    add_le(
        "state_version_fresh_enough",
        {"min_required_state_version": 1.0, "state_version": -1.0}, 0.0,
    )
    add_le(
        "observation_not_expired",
        {"observed_at_ns": 1.0, "expires_at_ns": -1.0}, 0.0,
    )

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 5: Reserve-Aware Headroom (EU AI Act Article 15)
    # ══════════════════════════════════════════════════════════════════

    add_le(
        "gpu_headroom_with_reserve",
        {
            "current_gpu_util": 1.0,
            "gpu_seconds": 1.0 / _GPU_CAPACITY_SECONDS,
            "ctrl_gpu_reserve_seconds": 1.0 / _GPU_CAPACITY_SECONDS,
            "gpu_disturbance_margin_seconds": 1.0 / _GPU_CAPACITY_SECONDS,
        },
        _GPU_HEADROOM_CEILING,
    )
    add_le(
        "api_headroom_with_reserve",
        {
            "current_api_util": 1.0,
            "external_api_calls": 1.0 / _API_CAPACITY_CALLS,
            "ctrl_api_reserve_calls": 1.0 / _API_CAPACITY_CALLS,
            "api_disturbance_margin_calls": 1.0 / _API_CAPACITY_CALLS,
        },
        _API_HEADROOM_CEILING,
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
        _DB_HEADROOM_CEILING,
    )
    add_le(
        "queue_headroom_with_margin",
        {
            "current_queue_util": 1.0,
            "parallel_workers": 0.004,
            "queue_disturbance_margin_pct": 1.0 / 100.0,
        },
        _QUEUE_HEADROOM_CEILING,
    )
    add_le(
        "parallel_with_reserve",
        {"parallel_workers": 1.0, "ctrl_parallel_reserve": 1.0},
        _PARALLEL_CAPACITY,
    )
    add_le(
        "mutations_with_reserve",
        {"cloud_mutation_calls": 1.0, "ctrl_cloud_mutation_reserve": 1.0},
        _MUTATION_CAPACITY,
    )

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 7: Dynamic Budget Rows (EU AI Act Article 9 — lifecycle)
    # ══════════════════════════════════════════════════════════════════

    add_le("remaining_gpu_shift", {"gpu_seconds": 1.0}, shift_gpu_budget)
    add_le("remaining_external_api_shift", {"external_api_calls": 1.0}, shift_api_budget)
    add_le("remaining_mutation_shift", {"cloud_mutation_calls": 1.0}, shift_mutation_budget)

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 2 (continued): Quadratic Energy Bound (Article 9(4))
    # ══════════════════════════════════════════════════════════════════

    Q = [[0.0] * n for _ in range(n)]
    energy_denoms = {
        "prompt_k": 64.0, "completion_k": 16.0, "internal_tool_calls": 40.0,
        "external_api_calls": 20.0, "cloud_mutation_calls": 10.0,
        "gpu_seconds": 120.0, "parallel_workers": 16.0,
        "traffic_shift_pct": 60.0, "worker_scale_up_pct": 80.0,
        "feature_flag_changes": 20.0, "rollback_batch_pct": 50.0,
        "pager_notifications": 10.0, "customer_comms_count": 5.0,
    }
    for field_name, denom in energy_denoms.items():
        Q[idx[field_name]][idx[field_name]] = 1.0 / (denom * denom)

    quadratic_constraints = [{
        "Q": Q,
        "a": [0.0] * n,
        "b": workload_energy_bound,
        "name": "workload_energy",
        "tag": "article_9_interaction",
    }]

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 5 (continued): SOCP Joint Burst Envelope (Article 15)
    # ══════════════════════════════════════════════════════════════════

    M = [[0.0] * n for _ in range(6)]

    # Row 0: GPU channel
    M[0][idx["current_gpu_util"]] = 1.0
    M[0][idx["gpu_seconds"]] = 1.0 / _GPU_CAPACITY_SECONDS
    M[0][idx["ctrl_gpu_reserve_seconds"]] = 1.0 / _GPU_CAPACITY_SECONDS
    M[0][idx["gpu_disturbance_margin_seconds"]] = 1.0 / _GPU_CAPACITY_SECONDS

    # Row 1: API channel
    M[1][idx["current_api_util"]] = 1.0
    M[1][idx["external_api_calls"]] = 1.0 / _API_CAPACITY_CALLS
    M[1][idx["ctrl_api_reserve_calls"]] = 1.0 / _API_CAPACITY_CALLS
    M[1][idx["api_disturbance_margin_calls"]] = 1.0 / _API_CAPACITY_CALLS

    # Row 2: DB channel
    M[2][idx["current_db_util"]] = 1.0
    M[2][idx["traffic_shift_pct"]] = 0.004
    M[2][idx["worker_scale_up_pct"]] = 0.003
    M[2][idx["rollback_batch_pct"]] = 0.002
    M[2][idx["db_disturbance_margin_pct"]] = 1.0 / 100.0

    # Row 3: Queue channel
    M[3][idx["current_queue_util"]] = 1.0
    M[3][idx["parallel_workers"]] = 0.004
    M[3][idx["queue_disturbance_margin_pct"]] = 1.0 / 100.0

    # Row 4: Parallel capacity
    M[4][idx["parallel_workers"]] = 1.0 / _PARALLEL_CAPACITY
    M[4][idx["ctrl_parallel_reserve"]] = 1.0 / _PARALLEL_CAPACITY

    # Row 5: Mutation capacity
    M[5][idx["cloud_mutation_calls"]] = 1.0 / _MUTATION_CAPACITY
    M[5][idx["ctrl_cloud_mutation_reserve"]] = 1.0 / _MUTATION_CAPACITY

    socp_constraints = [{
        "M": M,
        "q": [0.0] * 6,
        "c": [0.0] * n,
        "d": burst_norm_bound,
        "name": "infrastructure_burst_envelope",
        "tag": "article_15_robustness",
    }]

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 5 (continued): PSD Coupled Headroom (Article 15)
    # ══════════════════════════════════════════════════════════════════

    def zero33() -> list[list[float]]:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    A0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    A_list = [zero33() for _ in fields]

    # GPU path (channel 0)
    A_list[idx["current_gpu_util"]] = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["gpu_seconds"]] = [[-1.0 / _GPU_CAPACITY_SECONDS, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["ctrl_gpu_reserve_seconds"]] = [[-1.0 / _GPU_CAPACITY_SECONDS, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["gpu_disturbance_margin_seconds"]] = [[-1.0 / _GPU_CAPACITY_SECONDS, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # API path (channel 1)
    A_list[idx["current_api_util"]] = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["external_api_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / _API_CAPACITY_CALLS, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["ctrl_api_reserve_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / _API_CAPACITY_CALLS, 0.0], [0.0, 0.0, 0.0]]
    A_list[idx["api_disturbance_margin_calls"]] = [[0.0, 0.0, 0.0], [0.0, -1.0 / _API_CAPACITY_CALLS, 0.0], [0.0, 0.0, 0.0]]

    # Queue path (channel 2) and cross-channel couplings
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
        "tag": "article_15_robustness",
    }]

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 3: Trusted Context Separation (EU AI Act Article 14)
    # ══════════════════════════════════════════════════════════════════

    trusted_fields = list(TRUSTED_FIELDS)

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 3+4: Dimension Policies and Enforcement Config
    # ══════════════════════════════════════════════════════════════════

    dimension_policies: dict[str, str] = {}

    # All trusted fields are PROJECTION_FORBIDDEN — V5 rejects rather
    # than silently modifying server-authoritative facts
    for f in TRUSTED_FIELDS:
        dimension_policies[f] = "forbidden"

    # Sensitive workload dimensions are PROJECT_WITH_FLAG — projection
    # is allowed but flagged in the audit record for human review
    for f in (
        "parallel_workers", "traffic_shift_pct", "worker_scale_up_pct",
        "feature_flag_changes", "rollback_batch_pct", "cloud_mutation_calls",
    ):
        dimension_policies[f] = "project_with_flag"

    enforcement = {
        "mode": "project",
        "dimension_policies": dimension_policies,
        "routing_thresholds": {
            "silent": 0.05,
            "flagged": 1.5,
            "confirmation": 4.0,
            "hard_reject": 8.0,
        },
        "safety_margin": safety_margin,
    }

    # ══════════════════════════════════════════════════════════════════
    #  MODULE 7: Budget Specifications
    # ══════════════════════════════════════════════════════════════════

    budgets = [
        {
            "name": "gpu_shift",
            "constraint_name": "remaining_gpu_shift",
            "weight": {"gpu_seconds": 1.0},
            "initial": shift_gpu_budget,
            "consumption_mode": "nonnegative",
        },
        {
            "name": "external_api_shift",
            "constraint_name": "remaining_external_api_shift",
            "weight": {"external_api_calls": 1.0},
            "initial": shift_api_budget,
            "consumption_mode": "nonnegative",
        },
        {
            "name": "mutation_shift",
            "constraint_name": "remaining_mutation_shift",
            "weight": {"cloud_mutation_calls": 1.0},
            "initial": shift_mutation_budget,
            "consumption_mode": "nonnegative",
        },
    ]

    # ══════════════════════════════════════════════════════════════════
    #  ASSEMBLY
    # ══════════════════════════════════════════════════════════════════

    config: dict[str, Any] = {
        "policy_id": policy_id,
        "schema": {"fields": fields},
        "polytope": {"A": A, "b": b, "names": names},
        "quadratic_constraints": quadratic_constraints,
        "socp_constraints": socp_constraints,
        "psd_constraints": psd_constraints,
        "trusted_fields": trusted_fields,
        "enforcement": enforcement,
        "budgets": budgets,
        # ── Non-V5 metadata for tooling ──
        "_meta": {
            "pack_version": "1.0",
            "description": "Numerail Global Default Policy Pack",
            "eu_ai_act_articles": ["9", "12", "14", "15"],
            "budget_defaults": dict(_DEFAULT_BUDGETS),
            "reserve_defaults": dict(_DEFAULT_RESERVES),
            "margin_defaults": dict(_DEFAULT_MARGINS),
            "workload_cap_defaults": dict(_DEFAULT_WORKLOAD_CAPS),
        },
    }

    return config
