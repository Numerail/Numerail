"""Numerail Performance Benchmark Suite.

Standalone script — run with:
    python packages/numerail/tests/benchmark_performance.py

Measures latency (µs) and throughput (ops/sec) across the full enforcement
stack: core engine, individual primitives, enforcement modes, extension layer,
and scaling curves.

Methodology
-----------
- Timing: time.perf_counter_ns() around each call
- Runs: 1,000 per benchmark (sections 1–5); varies for section 6
- Warmup: first 100 runs discarded
- Stats: mean, median, p95, p99, min, max in microseconds
- Output: console table + BENCHMARK_REPORT.md in same directory
"""

from __future__ import annotations

import io
import os
import platform
import sys
import time

# Ensure stdout is UTF-8 on Windows (avoids cp1252 encoding errors for
# Unicode characters in headers and section separators).
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: ensure packages are importable when run from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in ("packages/numerail/src", "packages/numerail_ext/src"):
    _p = str(_REPO_ROOT / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from numerail.engine import (
    AuditChain,
    BudgetSpec,
    BudgetTracker,
    EnforcementConfig,
    EnforcementOutput,
    EnforcementResult,
    FeasibleRegion,
    LinearConstraints,
    Schema,
    box_constraints,
    combine_regions,
    enforce,
    halfplane,
    project,
)
from numerail.local import DefaultTimeProvider, NumerailSystemLocal
from numerail.parser import PolicyParser

_HAS_EXT = False
try:
    from numerail_ext.survivability.breaker import BreakerStateMachine
    from numerail_ext.survivability.global_default import build_global_default
    from numerail_ext.survivability.governor import StateTransitionGovernor
    from numerail_ext.survivability.local_backend import LocalNumerailBackend
    from numerail_ext.survivability.policy_builder import build_v5_policy_from_envelope
    from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
    from numerail_ext.survivability.types import (
        BreakerMode,
        BreakerThresholds,
        TelemetrySnapshot,
        WorkloadRequest,
    )
    from numerail_ext.survivability.contract import NumerailPolicyContract
    _HAS_EXT = True
except ImportError:
    pass

try:
    from numerail.parser import lint_config as _lint_config
    _HAS_LINT = True
except ImportError:
    _HAS_LINT = False


# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

RUNS = 1_000
WARMUP = 100


@dataclass
class BenchResult:
    name: str
    section: str
    n_runs: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    throughput_ops: Optional[float] = None  # ops/sec, if applicable
    note: str = ""


def _stats(samples_ns: List[int]) -> Tuple[float, float, float, float, float, float]:
    """Return (mean, median, p95, p99, min, max) in microseconds."""
    us = [s / 1_000.0 for s in samples_ns]
    us_sorted = sorted(us)
    n = len(us_sorted)
    p95 = us_sorted[int(0.95 * n)]
    p99 = us_sorted[int(0.99 * n)]
    return mean(us), median(us), p95, p99, us_sorted[0], us_sorted[-1]


def measure(
    fn: Callable,
    section: str,
    name: str,
    runs: int = RUNS,
    warmup: int = WARMUP,
    note: str = "",
) -> BenchResult:
    """Time fn() for `runs` iterations, discard first `warmup`, return stats."""
    samples: List[int] = []
    for i in range(runs + warmup):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        if i >= warmup:
            samples.append(t1 - t0)

    mn, med, p95, p99, lo, hi = _stats(samples)
    tput = 1_000_000.0 / mn if mn > 0 else float("inf")  # ops/sec
    return BenchResult(
        name=name,
        section=section,
        n_runs=runs,
        mean_us=mn,
        median_us=med,
        p95_us=p95,
        p99_us=p99,
        min_us=lo,
        max_us=hi,
        throughput_ops=tput,
        note=note,
    )


def measure_scaling(
    fn: Callable[[int], Callable],
    section: str,
    name_template: str,
    scale_values: List[int],
    runs: int = 500,
    warmup: int = 50,
    scale_label: str = "n",
) -> List[BenchResult]:
    """Benchmark fn(scale_value) across a range of scale values."""
    results = []
    for val in scale_values:
        bench_fn = fn(val)
        r = measure(bench_fn, section, name_template.format(**{scale_label: val}),
                    runs=runs, warmup=warmup, note=f"{scale_label}={val}")
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_linear_region(n: int, m_per_dim: int = 2) -> FeasibleRegion:
    """Box-style region in n dimensions with m_per_dim constraints per dim."""
    rng = np.random.default_rng(42)
    A_rows = []
    b_vals = []
    for i in range(n):
        for sign in [1.0, -1.0]:
            row = np.zeros(n)
            row[i] = sign
            A_rows.append(row)
            b_vals.append(1.0)
    # Add a few diagonal constraints to make it interesting
    for _ in range(min(n, 8)):
        w = rng.normal(size=n)
        w /= np.linalg.norm(w) + 1e-12
        A_rows.append(w)
        b_vals.append(0.9)
    A = np.array(A_rows)
    b = np.array(b_vals)
    names = [f"c{i}" for i in range(len(b_vals))]
    lc = LinearConstraints(A, b, names=names)
    return FeasibleRegion([lc], n)


def _make_box_region(n: int) -> FeasibleRegion:
    lower = -np.ones(n)
    upper = np.ones(n)
    return box_constraints(lower, upper)


def _make_schema(fields: List[str]) -> Schema:
    return Schema(fields)


def _feasible_point(region: FeasibleRegion, n: int) -> np.ndarray:
    """Return a point known to be feasible (origin for box regions)."""
    return np.zeros(n)


def _infeasible_point(n: int, scale: float = 2.0) -> np.ndarray:
    """Return a point outside a unit box."""
    x = np.ones(n) * scale
    return x


def _near_boundary_point(n: int, scale: float = 1.2) -> np.ndarray:
    """Return a point just outside the boundary — will project."""
    return np.ones(n) * scale


def _make_simple_policy(fields: List[str]) -> dict:
    n = len(fields)
    idx = {f: i for i, f in enumerate(fields)}
    A, b, names = [], [], []
    for f in fields:
        row = [0.0] * n
        row[idx[f]] = 1.0
        A.append(row)
        b.append(1000.0)
        names.append(f"max_{f}")
        row2 = [0.0] * n
        row2[idx[f]] = -1.0
        A.append(row2)
        b.append(0.0)
        names.append(f"min_{f}")
    return {
        "policy_id": "bench_policy",
        "schema": {"fields": fields},
        "polytope": {"A": A, "b": b, "names": names},
        "constraints": [],
        "budgets": [],
    }


def _make_local(fields: List[str]) -> NumerailSystemLocal:
    cfg = _make_simple_policy(fields)
    return NumerailSystemLocal(cfg, trusted_context_provider=None)


def _make_enforcement_output(n: int = 4) -> EnforcementOutput:
    """Create a minimal EnforcementOutput for AuditChain.append()."""
    region = _make_box_region(n)
    x = _feasible_point(region, n)
    return enforce(x, region)


if _HAS_EXT:
    def _make_snapshot(gpu=0.10, api=0.10, db=0.10, queue=0.10, err=0.0) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            state_version=1,
            observed_at_ns=time.time_ns(),
            current_gpu_util=gpu,
            current_api_util=api,
            current_db_util=db,
            current_queue_util=queue,
            current_error_rate_pct=err,
            ctrl_gpu_reserve_seconds=30.0,
            ctrl_api_reserve_calls=5.0,
            ctrl_parallel_reserve=4.0,
            ctrl_cloud_mutation_reserve=2.0,
            gpu_disturbance_margin_seconds=15.0,
            api_disturbance_margin_calls=3.0,
            db_disturbance_margin_pct=5.0,
            queue_disturbance_margin_pct=3.0,
        )

    _THRESHOLDS = BreakerThresholds(
        trip_score=0.50, reset_score=0.25, safe_stop_score=0.99
    )

    def _make_governor() -> StateTransitionGovernor:
        class _Res:
            def acquire(self, *, state_version, expires_at_ns, resource_claims): return "tok"
            def commit(self, *, token, receipt): pass
            def release(self, *, token): pass

        class _Dig:
            import hashlib, json
            def digest(self, payload):
                import hashlib, json
                return hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=str).encode()
                ).hexdigest()

        return StateTransitionGovernor(
            backend=LocalNumerailBackend(),
            transition_model=IncidentCommanderTransitionModel(freshness_ns=120_000_000_000),
            reservation_mgr=_Res(),
            digestor=_Dig(),
            thresholds=_THRESHOLDS,
            bootstrap_budgets={"gpu_shift": 500.0, "external_api_shift": 100.0, "mutation_shift": 40.0},
        )

    def _make_workload() -> WorkloadRequest:
        return WorkloadRequest(
            prompt_k=5.0, completion_k=2.0, internal_tool_calls=5.0,
            external_api_calls=3.0, cloud_mutation_calls=1.0, gpu_seconds=10.0,
            parallel_workers=2.0, traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
            feature_flag_changes=0.0, rollback_batch_pct=0.0,
            pager_notifications=1.0, customer_comms_count=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: CORE ENFORCEMENT LATENCY BY COMPLEXITY
# ═══════════════════════════════════════════════════════════════════════════

def run_section_1() -> List[BenchResult]:
    """Core enforce() latency across dimensionality and constraint complexity."""
    results = []
    SEC = "1. Core Enforcement Latency"

    # 1a. 2D box — trivial feasible path (APPROVE)
    r2 = _make_box_region(2)
    x2_ok = np.zeros(2)
    results.append(measure(lambda: enforce(x2_ok, r2), SEC, "1a. enforce 2D box (APPROVE)"))

    # 1b. 4D linear — APPROVE path
    r4 = _make_linear_region(4)
    x4_ok = np.zeros(4)
    results.append(measure(lambda: enforce(x4_ok, r4), SEC, "1b. enforce 4D linear (APPROVE)"))

    # 1c. 4D linear — PROJECT path (near boundary)
    x4_proj = _near_boundary_point(4, scale=1.5)
    results.append(measure(lambda: enforce(x4_proj, r4), SEC, "1c. enforce 4D linear (PROJECT)"))

    # 1d. 4D linear — REJECT path (far outside)
    cfg_reject = EnforcementConfig(mode="reject")
    x4_rej = _infeasible_point(4, scale=5.0)
    results.append(measure(lambda: enforce(x4_rej, r4, config=cfg_reject), SEC,
                            "1d. enforce 4D linear (REJECT mode, infeasible)"))

    # 1e. 8D linear — APPROVE
    r8 = _make_linear_region(8)
    x8_ok = np.zeros(8)
    results.append(measure(lambda: enforce(x8_ok, r8), SEC, "1e. enforce 8D linear (APPROVE)"))

    # 1f. 8D linear — PROJECT
    x8_proj = _near_boundary_point(8, scale=1.5)
    results.append(measure(lambda: enforce(x8_proj, r8), SEC, "1f. enforce 8D linear (PROJECT)"))

    # 1g. 16D linear — APPROVE
    r16 = _make_linear_region(16)
    x16_ok = np.zeros(16)
    results.append(measure(lambda: enforce(x16_ok, r16), SEC, "1g. enforce 16D linear (APPROVE)"))

    # 1h. 16D linear — PROJECT
    x16_proj = _near_boundary_point(16, scale=1.5)
    results.append(measure(lambda: enforce(x16_proj, r16), SEC, "1h. enforce 16D linear (PROJECT)"))

    # 1i. 30D linear — APPROVE
    r30 = _make_linear_region(30)
    x30_ok = np.zeros(30)
    results.append(measure(lambda: enforce(x30_ok, r30), SEC, "1i. enforce 30D linear (APPROVE)"))

    # 1j. 30D linear — PROJECT
    x30_proj = _near_boundary_point(30, scale=1.5)
    results.append(measure(lambda: enforce(x30_proj, r30), SEC, "1j. enforce 30D linear (PROJECT)"))

    # 1k. 8D with quadratic constraint (combined region)
    try:
        from numerail.engine import QuadraticConstraint
        Q = np.eye(8) * 2.0
        a = np.zeros(8)
        qc = QuadraticConstraint(Q, a, b=4.0, constraint_name="ellipsoid")
        lc8 = LinearConstraints(
            np.vstack([np.eye(8), -np.eye(8)]),
            np.ones(16),
            names=[f"c{i}" for i in range(16)],
        )
        r8q = FeasibleRegion([lc8, qc], 8)
        x8q_ok = np.zeros(8)
        results.append(measure(lambda: enforce(x8q_ok, r8q), SEC,
                                "1k. enforce 8D linear+quadratic (APPROVE)"))
        x8q_proj = np.ones(8) * 0.8
        results.append(measure(lambda: enforce(x8q_proj, r8q), SEC,
                                "1l. enforce 8D linear+quadratic (PROJECT)"))
    except Exception as e:
        print(f"  [skip 1k/1l: {e}]")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: INDIVIDUAL FUNCTION LATENCY
# ═══════════════════════════════════════════════════════════════════════════

def run_section_2() -> List[BenchResult]:
    """Latency of individual engine primitives in isolation."""
    results = []
    SEC = "2. Individual Function Latency"

    n = 8
    region = _make_linear_region(n)
    box = _make_box_region(n)
    x_ok = np.zeros(n)
    x_bad = np.ones(n) * 3.0
    x_proj = np.ones(n) * 1.3

    # 2a. FeasibleRegion.is_feasible — feasible point
    results.append(measure(lambda: region.is_feasible(x_ok), SEC,
                            "2a. FeasibleRegion.is_feasible (feasible)"))

    # 2b. FeasibleRegion.is_feasible — infeasible point
    results.append(measure(lambda: region.is_feasible(x_bad), SEC,
                            "2b. FeasibleRegion.is_feasible (infeasible)"))

    # 2c. project() standalone — fast convergence
    results.append(measure(lambda: project(x_proj, box), SEC,
                            "2c. project() box 8D (fast convergence)"))

    # 2d. project() standalone — linear region (more iterations)
    results.append(measure(lambda: project(x_proj, region), SEC,
                            "2d. project() linear 8D"))

    # 2e. Schema.vectorize()
    fields = [f"f{i}" for i in range(n)]
    schema = _make_schema(fields)
    values = {f: float(i) * 0.1 for i, f in enumerate(fields)}
    results.append(measure(lambda: schema.vectorize(values), SEC,
                            "2e. Schema.vectorize() 8 fields"))

    # 2f. Schema.devectorize()
    vec = np.array([float(i) * 0.1 for i in range(n)])
    results.append(measure(lambda: schema.devectorize(vec), SEC,
                            "2f. Schema.devectorize() 8 fields"))

    # 2g. BudgetTracker.record_consumption()
    tracker = BudgetTracker()
    spec = BudgetSpec(
        name="gpu_shift",
        constraint_name="max_f0",
        dimension_name="f0",
        weight=1.0,
        initial=1000.0,
        consumption_mode="nonnegative",
    )
    tracker.register(spec)

    # Build a minimal region/schema for the tracker
    r_budget = _make_box_region(n)
    schema_budget = Schema(fields)
    enforced_vec = np.zeros(n)

    _budget_counter = [0]
    def _consume():
        _budget_counter[0] += 1
        tracker.record_consumption(enforced_vec, f"act_{_budget_counter[0]}", schema_budget)

    results.append(measure(_consume, SEC, "2g. BudgetTracker.record_consumption()"))

    # 2h. BudgetTracker.rollback()
    # Pre-populate with actions to roll back
    tracker2 = BudgetTracker()
    tracker2.register(spec)
    for i in range(200):
        tracker2.record_consumption(enforced_vec, f"pre_{i}", schema_budget)

    _rb_counter = [0]
    def _rollback():
        _rb_counter[0] = (_rb_counter[0] + 1) % 200
        tracker2.rollback(f"pre_{_rb_counter[0]}")

    results.append(measure(_rollback, SEC, "2h. BudgetTracker.rollback()"))

    # 2i. AuditChain.append()
    chain = AuditChain()
    output = _make_enforcement_output(n)
    results.append(measure(lambda: chain.append(output), SEC,
                            "2i. AuditChain.append() (growing chain)"))

    # 2j. AuditChain.verify() — 100 records
    chain100 = AuditChain()
    output100 = _make_enforcement_output(4)
    for _ in range(100):
        chain100.append(output100)
    results.append(measure(lambda: chain100.verify(), SEC,
                            "2j. AuditChain.verify() 100 records",
                            runs=200, warmup=20))

    # 2k. AuditChain.verify() — 500 records
    chain500 = AuditChain()
    for _ in range(500):
        chain500.append(output100)
    results.append(measure(lambda: chain500.verify(), SEC,
                            "2k. AuditChain.verify() 500 records",
                            runs=100, warmup=10))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: ENFORCEMENT MODES
# ═══════════════════════════════════════════════════════════════════════════

def run_section_3() -> List[BenchResult]:
    """Latency breakdown by enforcement mode and decision path."""
    results = []
    SEC = "3. Enforcement Modes"

    n = 8
    region = _make_linear_region(n)
    x_ok = np.zeros(n)
    x_proj = np.ones(n) * 1.3
    x_far = np.ones(n) * 10.0

    cfg_project = EnforcementConfig(mode="project")
    cfg_reject = EnforcementConfig(mode="reject")
    cfg_hybrid = EnforcementConfig(mode="hybrid", max_distance=2.0)

    # 3a. Default mode (project) — feasible → APPROVE
    results.append(measure(lambda: enforce(x_ok, region), SEC,
                            "3a. default mode — APPROVE (feasible input)"))

    # 3b. Default mode — PROJECT path
    results.append(measure(lambda: enforce(x_proj, region), SEC,
                            "3b. default mode — PROJECT (infeasible input)"))

    # 3c. Reject mode — REJECT path
    results.append(measure(lambda: enforce(x_proj, region, config=cfg_reject), SEC,
                            "3c. reject mode — REJECT (infeasible input)"))

    # 3d. Project mode — explicit config, feasible
    results.append(measure(lambda: enforce(x_ok, region, config=cfg_project), SEC,
                            "3d. project mode — APPROVE (feasible input)"))

    # 3e. Project mode — explicit config, project
    results.append(measure(lambda: enforce(x_proj, region, config=cfg_project), SEC,
                            "3e. project mode — PROJECT (infeasible input)"))

    # 3f. Hybrid mode — feasible (APPROVE)
    results.append(measure(lambda: enforce(x_ok, region, config=cfg_hybrid), SEC,
                            "3f. hybrid mode — APPROVE (feasible input)"))

    # 3g. Hybrid mode — near boundary (PROJECT)
    results.append(measure(lambda: enforce(x_proj, region, config=cfg_hybrid), SEC,
                            "3g. hybrid mode — PROJECT (near boundary)"))

    # 3h. Hard REJECT — far outside, all modes give REJECT before solver
    results.append(measure(lambda: enforce(x_far, region), SEC,
                            "3h. default mode — REJECT (far outside, hard wall)"))

    # 3i. With schema (vectorize overhead)
    fields = [f"f{i}" for i in range(n)]
    schema = Schema(fields)
    results.append(measure(lambda: enforce(x_ok, region, schema=schema), SEC,
                            "3i. enforce with schema (8 fields) — APPROVE"))

    results.append(measure(lambda: enforce(x_proj, region, schema=schema), SEC,
                            "3j. enforce with schema (8 fields) — PROJECT"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: EXTENSION LATENCY (numerail_ext)
# ═══════════════════════════════════════════════════════════════════════════

def run_section_4() -> List[BenchResult]:
    """Extension-layer latency: breaker, governor, contract, parser."""
    results = []
    SEC = "4. Extension Layer Latency"

    if not _HAS_EXT:
        print("  [skip section 4: numerail_ext not installed]")
        return results

    # 4a. BreakerStateMachine.update() — healthy telemetry → CLOSED
    snap_healthy = _make_snapshot(0.10, 0.10, 0.10, 0.10, 0.0)
    breaker = BreakerStateMachine(_THRESHOLDS)
    results.append(measure(lambda: breaker.update(snap_healthy), SEC,
                            "4a. BreakerStateMachine.update() — CLOSED (healthy)"))

    # 4b. BreakerStateMachine.update() — high load → THROTTLED
    snap_high = _make_snapshot(0.80, 0.80, 0.80, 0.80, 0.0)
    breaker2 = BreakerStateMachine(_THRESHOLDS)
    results.append(measure(lambda: breaker2.update(snap_high), SEC,
                            "4b. BreakerStateMachine.update() — THROTTLED (high load)"))

    # 4c. synthesize_envelope() — CLOSED mode
    model = IncidentCommanderTransitionModel(freshness_ns=120_000_000_000)
    budgets = {"gpu_shift": 500.0, "external_api_shift": 100.0, "mutation_shift": 40.0}
    results.append(measure(
        lambda: model.synthesize_envelope(snapshot=snap_healthy, mode=BreakerMode.CLOSED, budgets=budgets),
        SEC, "4c. synthesize_envelope() — CLOSED mode",
    ))

    # 4d. synthesize_envelope() — THROTTLED mode
    results.append(measure(
        lambda: model.synthesize_envelope(snapshot=snap_high, mode=BreakerMode.THROTTLED, budgets=budgets),
        SEC, "4d. synthesize_envelope() — THROTTLED mode",
    ))

    # 4e. build_v5_policy_from_envelope()
    envelope = model.synthesize_envelope(snapshot=snap_healthy, mode=BreakerMode.CLOSED, budgets=budgets)
    results.append(measure(
        lambda: build_v5_policy_from_envelope(envelope),
        SEC, "4e. build_v5_policy_from_envelope()",
    ))

    # 4f. PolicyParser.parse() — simple 4-field config
    parser = PolicyParser()
    simple_cfg = _make_simple_policy(["amount", "rate", "duration", "risk_score"])
    results.append(measure(lambda: parser.parse(simple_cfg), SEC,
                            "4f. PolicyParser.parse() — 4-field config"))

    # 4g. PolicyParser.parse() — complex 8-field config
    complex_cfg = _make_simple_policy([f"field_{i}" for i in range(8)])
    results.append(measure(lambda: parser.parse(complex_cfg), SEC,
                            "4g. PolicyParser.parse() — 8-field config"))

    # 4h. Full governor step — enforce_next_step (CLOSED, approve path)
    governor = _make_governor()
    workload = _make_workload()
    snap_fresh = _make_snapshot(0.10, 0.10, 0.10, 0.10, 0.0)

    _gov_counter = [0]
    def _gov_step():
        _gov_counter[0] += 1
        governor.enforce_next_step(
            request=workload,
            snapshot=snap_fresh,
            action_id=f"bench_{_gov_counter[0]}",
        )

    results.append(measure(_gov_step, SEC,
                            "4h. governor.enforce_next_step() — CLOSED (healthy)",
                            runs=500, warmup=50))

    # 4i. NumerailPolicyContract.from_v5_config() — content-addressing
    try:
        global_cfg = build_global_default()
        results.append(measure(
            lambda: NumerailPolicyContract.from_v5_config(
                config=global_cfg,
                author_id="bench",
                policy_id="global::v1",
                scope="benchmark",
            ),
            SEC, "4i. NumerailPolicyContract.from_v5_config()",
            runs=500, warmup=50,
        ))

        # 4j. contract.verify_digest()
        contract = NumerailPolicyContract.from_v5_config(
            config=global_cfg, author_id="bench", policy_id="global::v1", scope="benchmark"
        )
        results.append(measure(lambda: contract.verify_digest(), SEC,
                                "4j. contract.verify_digest()"))
    except Exception as e:
        print(f"  [skip 4i/4j: {e}]")

    # 4k. lint_config()
    if _HAS_LINT:
        lint_cfg = _make_simple_policy(["amount", "rate"])
        results.append(measure(lambda: _lint_config(lint_cfg), SEC,
                                "4k. lint_config() — 2-field policy"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: THROUGHPUT
# ═══════════════════════════════════════════════════════════════════════════

def run_section_5() -> List[BenchResult]:
    """Sustained throughput under realistic call patterns."""
    results = []
    SEC = "5. Throughput"

    # 5a. NumerailSystemLocal.enforce() — full stack, simple policy
    local2 = _make_local(["amount", "rate"])
    _c2 = [0]
    def _local2_enforce():
        _c2[0] += 1
        local2.enforce({"amount": 100.0, "rate": 0.05}, action_id=f"b_{_c2[0]}")

    results.append(measure(_local2_enforce, SEC,
                            "5a. NumerailSystemLocal.enforce() — 2 fields (full stack)"))

    # 5b. NumerailSystemLocal.enforce() — 8-field policy
    local8 = _make_local([f"f{i}" for i in range(8)])
    _c8 = [0]
    def _local8_enforce():
        _c8[0] += 1
        local8.enforce({f"f{i}": float(i) * 0.1 for i in range(8)}, action_id=f"b_{_c8[0]}")

    results.append(measure(_local8_enforce, SEC,
                            "5b. NumerailSystemLocal.enforce() — 8 fields (full stack)"))

    # 5c. enforce() standalone — 4D box (minimal overhead)
    r4 = _make_box_region(4)
    x4 = np.zeros(4)
    results.append(measure(lambda: enforce(x4, r4), SEC,
                            "5c. enforce() standalone — 4D box (APPROVE)"))

    # 5d. enforce() standalone — 16D linear (project path)
    r16 = _make_linear_region(16)
    x16_proj = np.ones(16) * 1.3
    results.append(measure(lambda: enforce(x16_proj, r16), SEC,
                            "5d. enforce() standalone — 16D linear (PROJECT)"))

    # 5e. enforce() + AuditChain.append() combined throughput
    r8 = _make_linear_region(8)
    x8 = np.zeros(8)
    chain_tput = AuditChain()
    def _enforce_and_audit():
        out = enforce(x8, r8)
        chain_tput.append(out)

    results.append(measure(_enforce_and_audit, SEC,
                            "5e. enforce() + AuditChain.append() combined"))

    # 5f. Full local stack with audit access
    local_full = _make_local(["amount", "rate", "duration", "risk"])
    _cf = [0]
    def _full_cycle():
        _cf[0] += 1
        result = local_full.enforce(
            {"amount": 100.0, "rate": 0.05, "duration": 30.0, "risk": 0.3},
            action_id=f"fc_{_cf[0]}",
        )
        _ = result["decision"]

    results.append(measure(_full_cycle, SEC,
                            "5f. NumerailSystemLocal full cycle (enforce + result access)"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: SCALING CURVES
# ═══════════════════════════════════════════════════════════════════════════

def run_section_6() -> List[BenchResult]:
    """How latency scales with constraint count, dimension count, chain length."""
    results = []
    SEC = "6. Scaling Curves"

    # 6a. Constraint count scaling (dimension fixed at 8)
    def _make_constraint_bench(n_constraints: int) -> Callable:
        n_dim = 8
        rng = np.random.default_rng(n_constraints)
        A_rows, b_vals = [], []
        # Box constraints first
        for i in range(n_dim):
            r = np.zeros(n_dim); r[i] = 1.0
            A_rows.append(r); b_vals.append(1.0)
            r2 = np.zeros(n_dim); r2[i] = -1.0
            A_rows.append(r2); b_vals.append(1.0)
        # Additional random halfplane constraints
        extra = max(0, n_constraints - 2 * n_dim)
        for _ in range(extra):
            w = rng.normal(size=n_dim)
            w /= np.linalg.norm(w) + 1e-12
            A_rows.append(w); b_vals.append(0.9)
        A = np.array(A_rows[:n_constraints] if len(A_rows) > n_constraints else A_rows)
        b = np.array(b_vals[:n_constraints] if len(b_vals) > n_constraints else b_vals)
        lc = LinearConstraints(A, b, names=[f"c{i}" for i in range(len(b))])
        region = FeasibleRegion([lc], n_dim)
        x_proj = np.ones(n_dim) * 1.3
        return lambda: enforce(x_proj, region)

    constraint_counts = [4, 8, 16, 32, 64, 128]
    for nc in constraint_counts:
        fn = _make_constraint_bench(nc)
        r = measure(fn, SEC, f"6a. enforce — {nc} constraints, 8D (PROJECT)",
                    runs=500, warmup=50, note=f"n_constraints={nc}")
        results.append(r)

    # 6b. Dimension count scaling (constraints = 2*n, box-style)
    def _make_dim_bench(n_dim: int) -> Callable:
        region = _make_linear_region(n_dim)
        x_proj = np.ones(n_dim) * 1.3
        return lambda: enforce(x_proj, region)

    dim_counts = [2, 4, 8, 16, 30, 64]
    for nd in dim_counts:
        fn = _make_dim_bench(nd)
        r = measure(fn, SEC, f"6b. enforce — 8D {nd}D linear (PROJECT)",
                    runs=500, warmup=50, note=f"n_dim={nd}")
        results.append(r)

    # 6c. AuditChain.verify() scaling by chain length
    def _make_verify_bench(n_records: int) -> Callable:
        chain = AuditChain()
        out = _make_enforcement_output(4)
        for _ in range(n_records):
            chain.append(out)
        return lambda: chain.verify()

    chain_lengths = [10, 50, 100, 250, 500, 1000]
    for cl in chain_lengths:
        fn = _make_verify_bench(cl)
        n_runs = max(50, min(500, 5000 // cl))
        r = measure(fn, SEC, f"6c. AuditChain.verify() — {cl} records",
                    runs=n_runs, warmup=max(5, n_runs // 10), note=f"chain_length={cl}")
        results.append(r)

    # 6d. BudgetTracker scaling by number of registered budgets
    def _make_budget_bench(n_budgets: int) -> Callable:
        fields = [f"f{i}" for i in range(max(n_budgets, 2))]
        schema = Schema(fields)
        tracker = BudgetTracker()
        for i in range(n_budgets):
            tracker.register(BudgetSpec(
                name=f"budget_{i}",
                constraint_name=f"max_f{i % len(fields)}",
                dimension_name=fields[i % len(fields)],
                weight=1.0,
                initial=1000.0,
                consumption_mode="nonnegative",
            ))
        vec = np.zeros(len(fields))
        _counter = [0]
        def _consume():
            _counter[0] += 1
            tracker.record_consumption(vec, f"act_{_counter[0]}", schema)
        return _consume

    budget_counts = [1, 2, 4, 8]
    for nb in budget_counts:
        fn = _make_budget_bench(nb)
        r = measure(fn, SEC, f"6d. BudgetTracker.record_consumption() — {nb} budgets",
                    runs=500, warmup=50, note=f"n_budgets={nb}")
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v: float) -> str:
    if v >= 1000:
        return f"{v:>10,.1f}"
    elif v >= 10:
        return f"{v:>10.1f}"
    elif v >= 1:
        return f"{v:>10.2f}"
    else:
        return f"{v:>10.4f}"


def _print_table(results: List[BenchResult]) -> None:
    header = f"{'Benchmark':<58} {'mean':>8} {'median':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}  {'ops/sec':>12}"
    sep = "\u2500" * len(header)
    print(sep)
    print(header)
    _us = "\u00b5s"
    print(f"{'':58} {_us:>8} {_us:>8} {_us:>8} {_us:>8} {_us:>8} {_us:>8}  {'':>12}")
    print(sep)

    current_section = None
    for r in results:
        if r.section != current_section:
            if current_section is not None:
                print()
            print(f"\n  \u2500\u2500 {r.section} \u2500\u2500")
            current_section = r.section

        tput_str = f"{r.throughput_ops:>12,.0f}" if r.throughput_ops else ""
        print(
            f"  {r.name:<56}"
            f"{_fmt(r.mean_us)}"
            f"{_fmt(r.median_us)}"
            f"{_fmt(r.p95_us)}"
            f"{_fmt(r.p99_us)}"
            f"{_fmt(r.min_us)}"
            f"{_fmt(r.max_us)}"
            f"  {tput_str}"
        )
    print(sep)


def _platform_info() -> dict:
    import platform
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "runs_per_bench": RUNS,
        "warmup_discarded": WARMUP,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _group_by_section(results: List[BenchResult]) -> Dict[str, List[BenchResult]]:
    grouped: Dict[str, List[BenchResult]] = {}
    for r in results:
        grouped.setdefault(r.section, []).append(r)
    return grouped


def _md_table(results: List[BenchResult]) -> str:
    rows = ["| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |",
            "|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|"]
    for r in results:
        tput = f"{r.throughput_ops:,.0f}" if r.throughput_ops else "—"
        rows.append(
            f"| {r.name} "
            f"| {r.mean_us:.2f} "
            f"| {r.median_us:.2f} "
            f"| {r.p95_us:.2f} "
            f"| {r.p99_us:.2f} "
            f"| {r.min_us:.2f} "
            f"| {r.max_us:.2f} "
            f"| {tput} |"
        )
    return "\n".join(rows)


def _deployment_corridor(results: List[BenchResult]) -> str:
    """Identify p99 latency budget for typical deployment scenarios."""
    lookup = {r.name: r for r in results}

    def _find(prefix: str) -> Optional[BenchResult]:
        for r in results:
            if r.name.startswith(prefix):
                return r
        return None

    lines = []
    lines.append("### Deployment Latency Corridor")
    lines.append("")
    lines.append("Estimated end-to-end latency budget per enforcement call in a")
    lines.append("production deployment (p99 components, additive model):")
    lines.append("")
    lines.append("| Component | p99 µs | Notes |")
    lines.append("|-----------|-------:|-------|")

    components = [
        ("1b. enforce 4D linear (APPROVE)", "4D approve (policy load cached)"),
        ("1f. enforce 8D linear (PROJECT)", "8D project (correction path)"),
        ("2i. AuditChain.append() (growing chain)", "Audit record append"),
        ("2g. BudgetTracker.record_consumption()", "Budget delta write"),
        ("5a. NumerailSystemLocal.enforce() — 2 fields (full stack)", "Full local stack (2 fields)"),
        ("5b. NumerailSystemLocal.enforce() — 8 fields (full stack)", "Full local stack (8 fields)"),
    ]
    total = 0.0
    for name, note in components:
        r = _find(name)
        if r:
            lines.append(f"| {note} | {r.p99_us:.1f} | — |")
            total += r.p99_us

    lines.append(f"| **Estimated full-stack p99** | **{total:.1f}** | Additive model |")
    lines.append("")
    lines.append("*Note: real production adds network, DB transaction, and outbox overhead.*")
    return "\n".join(lines)


def _bottleneck_analysis(results: List[BenchResult]) -> str:
    """Identify the top 5 highest-latency operations."""
    sorted_results = sorted(results, key=lambda r: r.p99_us, reverse=True)
    lines = ["### Bottleneck Analysis", "", "Top 10 highest p99 operations:"]
    lines.append("")
    lines.append("| Rank | Benchmark | p99 µs | mean µs |")
    lines.append("|------|-----------|-------:|--------:|")
    for i, r in enumerate(sorted_results[:10], 1):
        lines.append(f"| {i} | {r.name} | {r.p99_us:.1f} | {r.mean_us:.1f} |")
    lines.append("")
    lines.append("**Lowest-latency operations (fast path):**")
    lines.append("")
    lines.append("| Rank | Benchmark | mean µs | ops/sec |")
    lines.append("|------|-----------|--------:|--------:|")
    sorted_asc = sorted(results, key=lambda r: r.mean_us)
    for i, r in enumerate(sorted_asc[:5], 1):
        tput = f"{r.throughput_ops:,.0f}" if r.throughput_ops else "—"
        lines.append(f"| {i} | {r.name} | {r.mean_us:.2f} | {tput} |")
    return "\n".join(lines)


def generate_report(results: List[BenchResult], out_path: Path) -> None:
    pinfo = _platform_info()
    grouped = _group_by_section(results)

    # Executive summary
    all_means = [r.mean_us for r in results]
    all_p99 = [r.p99_us for r in results]
    fastest = min(results, key=lambda r: r.mean_us)
    slowest = max(results, key=lambda r: r.mean_us)

    lines = []
    lines.append("# Numerail Performance Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("")
    lines.append("## Platform")
    lines.append("")
    lines.append(f"- **Python**: {pinfo['python']}")
    lines.append(f"- **Platform**: {pinfo['platform']}")
    lines.append(f"- **CPU**: {pinfo['cpu']}")
    lines.append(f"- **NumPy**: {pinfo['numpy']}")
    lines.append(f"- **Runs per benchmark**: {pinfo['runs_per_bench']} (first {pinfo['warmup_discarded']} discarded as warmup)")
    lines.append(f"- **numerail_ext available**: {'yes' if _HAS_EXT else 'no'}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total benchmarks**: {len(results)}")
    lines.append(f"- **Fastest operation**: `{fastest.name}` — {fastest.mean_us:.2f} µs mean")
    lines.append(f"- **Slowest operation**: `{slowest.name}` — {slowest.mean_us:.1f} µs mean")
    lines.append(f"- **Median mean latency across all benchmarks**: {median(all_means):.1f} µs")
    lines.append(f"- **Median p99 latency across all benchmarks**: {median(all_p99):.1f} µs")
    lines.append("")

    # Find key throughput numbers
    tput_results = [r for r in results if r.throughput_ops and r.throughput_ops > 0]
    if tput_results:
        best_tput = max(tput_results, key=lambda r: r.throughput_ops)
        local_tput = next((r for r in results if "NumerailSystemLocal" in r.name and "2 field" in r.name), None)
        lines.append("### Key Throughput Numbers")
        lines.append("")
        if local_tput:
            lines.append(f"- **NumerailSystemLocal full stack (2 fields)**: {local_tput.throughput_ops:,.0f} ops/sec")
        lines.append(f"- **Peak throughput (fastest path)**: {best_tput.throughput_ops:,.0f} ops/sec (`{best_tput.name}`)")
        lines.append("")

    lines.append(_deployment_corridor(results))
    lines.append("")
    lines.append(_bottleneck_analysis(results))
    lines.append("")

    # Per-section tables
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("All latencies in **µs (microseconds)**. Timing: `time.perf_counter_ns()`.")
    lines.append("")

    for section_name, section_results in grouped.items():
        lines.append(f"### {section_name}")
        lines.append("")
        lines.append(_md_table(section_results))
        lines.append("")

    # Scaling analysis
    lines.append("## Scaling Analysis")
    lines.append("")

    s6_results = grouped.get("6. Scaling Curves", [])
    if s6_results:
        # Constraint count scaling
        c_scale = [r for r in s6_results if "6a." in r.name]
        if c_scale:
            lines.append("### Constraint Count Scaling (8D, PROJECT path)")
            lines.append("")
            lines.append("| Constraints | mean µs | p99 µs | Relative to 4-constraint |")
            lines.append("|------------:|--------:|-------:|-------------------------:|")
            base = c_scale[0].mean_us if c_scale else 1.0
            for r in c_scale:
                rel = r.mean_us / base if base > 0 else 1.0
                nc = r.note.split("=")[-1] if r.note else "?"
                lines.append(f"| {nc} | {r.mean_us:.2f} | {r.p99_us:.2f} | {rel:.2f}× |")
            lines.append("")

        # Dimension scaling
        d_scale = [r for r in s6_results if "6b." in r.name]
        if d_scale:
            lines.append("### Dimension Count Scaling (PROJECT path)")
            lines.append("")
            lines.append("| Dimensions | mean µs | p99 µs | Relative to 2D |")
            lines.append("|----------:|--------:|-------:|---------------:|")
            base = d_scale[0].mean_us if d_scale else 1.0
            for r in d_scale:
                rel = r.mean_us / base if base > 0 else 1.0
                nd = r.note.split("=")[-1] if r.note else "?"
                lines.append(f"| {nd} | {r.mean_us:.2f} | {r.p99_us:.2f} | {rel:.2f}× |")
            lines.append("")

        # Chain length scaling
        cl_scale = [r for r in s6_results if "6c." in r.name]
        if cl_scale:
            lines.append("### AuditChain.verify() Scaling")
            lines.append("")
            lines.append("| Records | mean µs | p99 µs | µs per record |")
            lines.append("|--------:|--------:|-------:|--------------:|")
            for r in cl_scale:
                cl = int(r.note.split("=")[-1]) if r.note else 1
                per_rec = r.mean_us / cl if cl > 0 else 0.0
                lines.append(f"| {cl} | {r.mean_us:.2f} | {r.p99_us:.2f} | {per_rec:.4f} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `packages/numerail/tests/benchmark_performance.py`.*")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("  Numerail Performance Benchmark Suite")
    print(f"  Python {sys.version.split()[0]}  \u00b7  NumPy {np.__version__}")
    print(f"  {platform.platform()}")
    print(f"  Runs: {RUNS}  \u00b7  Warmup: {WARMUP}")
    if not _HAS_EXT:
        print("  numerail_ext: not installed - section 4 will be skipped")
    print("=" * 80)

    all_results: List[BenchResult] = []

    sections = [
        ("Section 1: Core Enforcement Latency", run_section_1),
        ("Section 2: Individual Function Latency", run_section_2),
        ("Section 3: Enforcement Modes", run_section_3),
        ("Section 4: Extension Layer Latency", run_section_4),
        ("Section 5: Throughput", run_section_5),
        ("Section 6: Scaling Curves", run_section_6),
    ]

    for title, fn in sections:
        print(f"\n  Running {title}…", flush=True)
        t0 = time.perf_counter()
        section_results = fn()
        elapsed = time.perf_counter() - t0
        all_results.extend(section_results)
        print(f"    {len(section_results)} benchmarks in {elapsed:.1f}s", flush=True)

    print()
    _print_table(all_results)

    report_path = Path(__file__).parent / "BENCHMARK_REPORT.md"
    generate_report(all_results, report_path)


if __name__ == "__main__":
    main()
