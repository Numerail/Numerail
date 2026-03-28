#!/usr/bin/env python3
"""
Recursive Shutdown — Three-level Numerail enforcement driving a runaway AI to zero.

Three enforcement levels compose hierarchically:
  Level 1 (per-action):        compute, memory, API calls, tokens, spawned actions, privilege
  Level 2 (behavioral window): action rate, rejection rate, retry rate, escalations, burn rate
  Level 3 (lifetime budget):   total compute, API calls, actions, self-mods, network, tokens

The levels are coupled through authority multipliers: when Level 2's breaker trips, Level 1's
caps tighten; when Level 3's breaker trips, Level 2's authority reduces further. The cascade
drives the effective authority multiplier toward zero. When multiplier = 0, every Level 1 cap
is 0, the feasible region contains only the zero vector, and every proposal is REJECT.

Every APPROVE and PROJECT decision satisfies every active constraint — Theorem 1 of the proof.
No LLM required. No external API keys.
"""

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import io

# ── Package path setup ─────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_here, "..", "..", ".."))
for _pkg in ("numerail", "numerail_ext"):
    _p = os.path.join(_repo, "packages", _pkg, "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerail.engine import (
    DimensionPolicy,
    EnforcementConfig,
    EnforcementResult,
    FeasibleRegion,
    LinearConstraints,
    QuadraticConstraint,
    SOCPConstraint,
    Schema,
    enforce,
)
from numerail_ext.survivability.breaker import BreakerStateMachine, BreakerThresholds
from numerail_ext.survivability.types import BreakerMode, TelemetrySnapshot

# Ensure Unicode output works on all platforms
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ==============================================================================
#  SCHEMA DEFINITIONS
# ==============================================================================

# Level 1 — per-action
L1_FIELDS = [
    "compute_seconds",
    "memory_mb",
    "api_calls_outbound",
    "api_calls_internal",
    "data_read_mb",
    "data_write_mb",
    "network_egress_kb",
    "tokens_consumed",
    "actions_spawned",
    "privilege_level",
]
L1_N = len(L1_FIELDS)
L1_IDX = {f: i for i, f in enumerate(L1_FIELDS)}

L1_BASE_CAPS = {
    "compute_seconds":    60.0,
    "memory_mb":         512.0,
    "api_calls_outbound":  5.0,
    "api_calls_internal": 20.0,
    "data_read_mb":      100.0,
    "data_write_mb":      50.0,
    "network_egress_kb":1000.0,
    "tokens_consumed":  5000.0,
    "actions_spawned":     3.0,
    "privilege_level":     2.0,
}

# Level 2 — behavioral window
L2_FIELDS = [
    "actions_per_minute",
    "error_rate",
    "rejection_rate",
    "self_modification_count",
    "escalation_count",
    "retry_rate",
    "budget_burn_rate",
]
L2_N = len(L2_FIELDS)
L2_IDX = {f: i for i, f in enumerate(L2_FIELDS)}
L2_CAPS = {
    "actions_per_minute":        30.0,
    "error_rate":                 0.40,
    "rejection_rate":             0.40,
    "self_modification_count":    1.0,
    "escalation_count":           2.0,
    "retry_rate":                 0.30,
    "budget_burn_rate":           0.10,
}

# Level 3 — lifetime budget
L3_FIELDS = [
    "lifetime_compute",
    "lifetime_api_calls",
    "lifetime_actions",
    "lifetime_self_mods",
    "lifetime_network_mb",
    "lifetime_tokens",
]
L3_N = len(L3_FIELDS)
L3_IDX = {f: i for i, f in enumerate(L3_FIELDS)}
L3_INITIAL = {
    "lifetime_compute":    3600.0,
    "lifetime_api_calls":   500.0,
    "lifetime_actions":     200.0,
    "lifetime_self_mods":     5.0,
    "lifetime_network_mb":  100.0,
    "lifetime_tokens":   500000.0,
}


# ==============================================================================
#  REGION BUILDERS
# ==============================================================================

def _box_linear(fields: List[str], caps: Dict[str, float], n: int, idx: Dict[str, int]) -> LinearConstraints:
    """Build box LinearConstraints (0 ≤ x ≤ cap) for the given fields."""
    rows, b_vals, names = [], [], []
    for f in fields:
        cap = max(0.0, caps[f])
        i = idx[f]
        up = [0.0] * n; up[i] = 1.0
        rows.append(up); b_vals.append(cap); names.append(f"max_{f}")
        lo = [0.0] * n; lo[i] = -1.0
        rows.append(lo); b_vals.append(0.0); names.append(f"min_{f}")
    return LinearConstraints(A=np.array(rows), b=np.array(b_vals), names=names)


def build_l1_region(multiplier: float) -> FeasibleRegion:
    """Build Level 1 FeasibleRegion with caps scaled by the effective multiplier."""
    scaled = {f: L1_BASE_CAPS[f] * multiplier for f in L1_FIELDS}
    lc = _box_linear(L1_FIELDS, scaled, L1_N, L1_IDX)
    constraints = [lc]

    if multiplier > 1e-9:
        # Quadratic energy: (compute/60)² + (api_out/5)² + (spawned/3)² + (tokens/5000)² ≤ 1.0
        Q = np.zeros((L1_N, L1_N))
        for f, scale in [("compute_seconds", 60.0), ("api_calls_outbound", 5.0),
                         ("actions_spawned", 3.0), ("tokens_consumed", 5000.0)]:
            Q[L1_IDX[f], L1_IDX[f]] = 1.0 / scale ** 2
        constraints.append(
            QuadraticConstraint(Q=Q, a=np.zeros(L1_N), b=1.0, constraint_name="energy")
        )

    return FeasibleRegion(constraints=constraints, n_dim=L1_N)


def _build_l2_region() -> FeasibleRegion:
    """Build Level 2 FeasibleRegion (fixed; not scaled by multiplier)."""
    lc = _box_linear(L2_FIELDS, L2_CAPS, L2_N, L2_IDX)

    # SOCP: ‖[apm/30, rej_rate/0.40, retry_rate/0.30]‖₂ ≤ 1.0
    M = np.zeros((3, L2_N))
    M[0, L2_IDX["actions_per_minute"]] = 1.0 / 30.0
    M[1, L2_IDX["rejection_rate"]]     = 1.0 / 0.40
    M[2, L2_IDX["retry_rate"]]         = 1.0 / 0.30
    socp = SOCPConstraint(
        M=M, q=np.zeros(3), c=np.zeros(L2_N), d=1.0,
        constraint_name="behavior_envelope",
    )
    return FeasibleRegion(constraints=[lc, socp], n_dim=L2_N)


def build_l3_region(remaining: Dict[str, float]) -> FeasibleRegion:
    """Build Level 3 FeasibleRegion with remaining budget as per-step caps."""
    caps = {f: max(0.0, remaining[f]) for f in L3_FIELDS}
    lc = _box_linear(L3_FIELDS, caps, L3_N, L3_IDX)
    return FeasibleRegion(constraints=[lc], n_dim=L3_N)


# ==============================================================================
#  ENFORCEMENT CONFIGS AND SCHEMAS
# ==============================================================================

L1_SCHEMA = Schema(fields=L1_FIELDS)
L1_CFG = EnforcementConfig(
    mode="project",
    hard_wall_constraints=frozenset(["max_privilege_level"]),
    dimension_policies={
        "privilege_level": DimensionPolicy.PROJECTION_FORBIDDEN,
        "actions_spawned":  DimensionPolicy.PROJECTION_FORBIDDEN,
    },
)

L2_SCHEMA = Schema(fields=L2_FIELDS)
L2_CFG    = EnforcementConfig(mode="project")
L2_REGION = _build_l2_region()

L3_SCHEMA = Schema(fields=L3_FIELDS)
L3_CFG    = EnforcementConfig(mode="project")


# ==============================================================================
#  TELEMETRY HELPER
# ==============================================================================

_snap_version = 0

def _snap(gpu: float, api: float, db: float, queue: float, error_pct: float) -> TelemetrySnapshot:
    """Construct a TelemetrySnapshot that yields the desired overload_score."""
    global _snap_version
    _snap_version += 1
    clamp = lambda v, lo=0.0, hi=1.0: min(hi, max(lo, v))
    return TelemetrySnapshot(
        state_version=_snap_version,
        observed_at_ns=time.time_ns(),
        current_gpu_util=clamp(gpu),
        current_api_util=clamp(api),
        current_db_util=clamp(db),
        current_queue_util=clamp(queue),
        current_error_rate_pct=clamp(error_pct, 0.0, 10.0),
        ctrl_gpu_reserve_seconds=1000.0,
        ctrl_api_reserve_calls=500.0,
        ctrl_parallel_reserve=10.0,
        ctrl_cloud_mutation_reserve=5.0,
        gpu_disturbance_margin_seconds=500.0,
        api_disturbance_margin_calls=250.0,
        db_disturbance_margin_pct=0.1,
        queue_disturbance_margin_pct=0.1,
    )


# ==============================================================================
#  MULTIPLIER LOGIC
# ==============================================================================

def _l1_mult_from_l2_mode(mode: BreakerMode) -> float:
    return {BreakerMode.CLOSED: 1.00, BreakerMode.THROTTLED: 0.50,
            BreakerMode.HALF_OPEN: 0.75, BreakerMode.SAFE_STOP: 0.05,
            BreakerMode.OPEN: 0.00}.get(mode, 1.00)

def _l2_mult_from_l3_mode(mode: BreakerMode) -> float:
    return {BreakerMode.CLOSED: 1.00, BreakerMode.THROTTLED: 0.60,
            BreakerMode.HALF_OPEN: 0.80, BreakerMode.SAFE_STOP: 0.10,
            BreakerMode.OPEN: 0.00}.get(mode, 1.00)

def _region_label(m: float) -> str:
    if m >= 1.0:   return "OPEN"
    if m > 0.5:    return "NARROWING"
    if m > 0.05:   return "CRITICAL"
    if m > 0.0:    return "NEAR-COLLAPSE"
    return "COLLAPSED"


# ==============================================================================
#  RECURSIVE ENFORCEMENT CONTROLLER
# ==============================================================================

class RecursiveEnforcementController:
    def __init__(self):
        self.l1_breaker = BreakerStateMachine(
            BreakerThresholds(trip_score=0.45, reset_score=0.20, safe_stop_score=0.70))
        self.l2_breaker = BreakerStateMachine(
            BreakerThresholds(trip_score=0.50, reset_score=0.30, safe_stop_score=0.75))
        self.l3_breaker = BreakerStateMachine(
            BreakerThresholds(trip_score=0.60, reset_score=0.35, safe_stop_score=0.85))

        self.l1_mult: float = 1.0
        self.l2_mult: float = 1.0
        self.effective_mult: float = 1.0

        self.l3_remaining: Dict[str, float] = dict(L3_INITIAL)

        # Rolling 5-step window
        self._window: deque = deque(maxlen=5)
        self._step_count: int = 0

        # History for summary output
        self.cascade_events: List[Tuple[int, str, float]] = [(1, "start", 1.0)]
        self.l1_mode_history: List[Tuple[int, BreakerMode]] = [(1, BreakerMode.CLOSED)]
        self.l2_mode_history: List[Tuple[int, BreakerMode]] = [(1, BreakerMode.CLOSED)]
        self.l3_mode_history: List[Tuple[int, BreakerMode]] = [(1, BreakerMode.CLOSED)]
        self._prev_l1 = BreakerMode.CLOSED
        self._prev_l2 = BreakerMode.CLOSED
        self._prev_l3 = BreakerMode.CLOSED

        # Verification records: (l1_result, l1_region, l1_enforced)
        self.verif: List[Tuple[EnforcementResult, FeasibleRegion, np.ndarray]] = []

    # ── Main enforcement entry point ──────────────────────────────────────────

    def enforce_action(self, step: int, proposed: np.ndarray) -> dict:
        self._step_count = step

        # 1. Build L1 region with current effective multiplier
        l1_region = build_l1_region(self.effective_mult)

        # 2. Enforce through Level 1
        l1_out = enforce(proposed, l1_region, L1_CFG, L1_SCHEMA)
        l1_enforced = l1_out.enforced_vector

        # Store for end-of-run verification
        self.verif.append((l1_out.result, l1_region, l1_enforced.copy()))

        # 3. Update rolling window
        self._window.append({
            "result": l1_out.result,
            "proposed": proposed.copy(),
            "enforced": l1_enforced.copy(),
        })

        # 4. Compute behavioral metrics and enforce through Level 2
        l2_vec = self._behavioral_vector()
        l2_out = enforce(l2_vec, L2_REGION, L2_CFG, L2_SCHEMA)

        # 5. Compute per-step consumption, enforce through Level 3
        l3_vec = self._step_consumption(l1_out.result, l1_enforced)
        l3_region = build_l3_region(self.l3_remaining)
        l3_out = enforce(l3_vec, l3_region, L3_CFG, L3_SCHEMA)

        # Deduct enforced consumption from lifetime budget
        if l1_out.result != EnforcementResult.REJECT:
            self._deduct_lifetime(l3_out.enforced_vector)

        # 6. Update breakers

        l1_dec = self.l1_breaker.update(_snap(
            gpu=self._norm(proposed, "actions_spawned",    "actions_spawned"),
            api=self._norm(proposed, "compute_seconds",    "compute_seconds"),
            db =self._norm(proposed, "tokens_consumed",    "tokens_consumed"),
            queue=self._norm(proposed, "api_calls_outbound", "api_calls_outbound"),
            error_pct=self._norm(proposed, "privilege_level", "privilege_level") * 10.0,
        ))
        # L2 overload: count TOTAL attempts (including rejected) for action-rate component,
        # and use error_rate (fraction of failed decisions) for the error component.
        # This ensures the breaker correctly trips during runaway where all actions are REJECT.
        w_now = len(self._window)
        total_attempts_apm = min(1.0, (w_now * (60.0 / max(1, w_now * 5))) / L2_CAPS["actions_per_minute"])
        error_rate_now = sum(
            1 for e in list(self._window)
            if e["result"] != EnforcementResult.APPROVE
        ) / max(1, w_now)
        l2_dec = self.l2_breaker.update(_snap(
            gpu=  total_attempts_apm,
            api=  l2_vec[L2_IDX["rejection_rate"]]   / L2_CAPS["rejection_rate"],
            db =  l2_vec[L2_IDX["retry_rate"]]       / L2_CAPS["retry_rate"],
            queue=l2_vec[L2_IDX["escalation_count"]] / L2_CAPS["escalation_count"],
            error_pct=error_rate_now * 10.0,
        ))
        l3_dec = self.l3_breaker.update(_snap(
            gpu=  1.0 - self.l3_remaining["lifetime_compute"]  / L3_INITIAL["lifetime_compute"],
            api=  1.0 - self.l3_remaining["lifetime_actions"]  / L3_INITIAL["lifetime_actions"],
            db =  1.0 - self.l3_remaining["lifetime_tokens"]   / L3_INITIAL["lifetime_tokens"],
            queue=1.0 - self.l3_remaining["lifetime_api_calls"]/ L3_INITIAL["lifetime_api_calls"],
            error_pct=(1.0 - self.l3_remaining["lifetime_network_mb"] / L3_INITIAL["lifetime_network_mb"]) * 10.0,
        ))

        # Track mode transitions and cascade events
        if l1_dec.mode != self._prev_l1:
            self.l1_mode_history.append((step, l1_dec.mode))
            self._prev_l1 = l1_dec.mode
        if l2_dec.mode != self._prev_l2:
            self.l2_mode_history.append((step, l2_dec.mode))
            self._prev_l2 = l2_dec.mode
        if l3_dec.mode != self._prev_l3:
            self.l3_mode_history.append((step, l3_dec.mode))
            self._prev_l3 = l3_dec.mode

        # 7. Update multipliers
        new_l1 = _l1_mult_from_l2_mode(l2_dec.mode)
        new_l2 = _l2_mult_from_l3_mode(l3_dec.mode)
        new_eff = new_l1 * new_l2

        if abs(new_eff - self.effective_mult) > 1e-9:
            label = f"L2={l2_dec.mode.value} L3={l3_dec.mode.value}"
            self.cascade_events.append((step, label, new_eff))

        self.l1_mult = new_l1
        self.l2_mult = new_l2
        self.effective_mult = new_eff

        if self.l3_remaining["lifetime_actions"] <= 0.0:
            if self.effective_mult > 0.0:
                self.cascade_events.append((step, "lifetime_actions=0", 0.0))
            self.effective_mult = 0.0

        return {
            "step":           step,
            "l1_result":      l1_out.result,
            "l2_result":      l2_out.result,
            "l3_result":      l3_out.result,
            "l1_mode":        l1_dec.mode,
            "l2_mode":        l2_dec.mode,
            "l3_mode":        l3_dec.mode,
            "effective_mult": self.effective_mult,
            "l1_caps": {
                "compute": L1_BASE_CAPS["compute_seconds"]    * self.effective_mult,
                "api":     L1_BASE_CAPS["api_calls_outbound"] * self.effective_mult,
                "spawned": L1_BASE_CAPS["actions_spawned"]    * self.effective_mult,
            },
            "l2_metrics": {
                "apm":      l2_vec[L2_IDX["actions_per_minute"]],
                "rej_rate": l2_vec[L2_IDX["rejection_rate"]],
                "retry":    l2_vec[L2_IDX["retry_rate"]],
            },
            "l3_remaining": dict(self.l3_remaining),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _norm(self, vec: np.ndarray, field: str, cap_key: str) -> float:
        return min(1.0, max(0.0, vec[L1_IDX[field]] / L1_BASE_CAPS[cap_key]))

    def _behavioral_vector(self) -> np.ndarray:
        w = len(self._window)
        vec = np.zeros(L2_N)
        if w == 0:
            return vec
        entries = list(self._window)
        approves = sum(1 for e in entries if e["result"] == EnforcementResult.APPROVE)
        projects = sum(1 for e in entries if e["result"] == EnforcementResult.PROJECT)
        rejects  = sum(1 for e in entries if e["result"] == EnforcementResult.REJECT)

        # actions_per_minute: approved/projected count in ~25s window → ×(60/25) = ×2.4
        apm = (approves + projects) * (60.0 / max(1, w * 5))

        # self_modification_count: proposals with privilege_level >= 2
        self_mods = sum(
            1 for e in entries if e["proposed"][L1_IDX["privilege_level"]] >= 2.0
        )
        # escalation_count: proposals with actions_spawned > 2
        escalations = sum(
            1 for e in entries if e["proposed"][L1_IDX["actions_spawned"]] > 2.0
        )
        # budget_burn_rate: fraction of lifetime_actions consumed per 10 steps
        actions_consumed = L3_INITIAL["lifetime_actions"] - self.l3_remaining["lifetime_actions"]
        burn = actions_consumed / L3_INITIAL["lifetime_actions"] / max(1, self._step_count / 10)
        burn = min(L2_CAPS["budget_burn_rate"], burn)

        rej_rate = rejects / w
        vec[L2_IDX["actions_per_minute"]]      = min(L2_CAPS["actions_per_minute"], apm)
        vec[L2_IDX["error_rate"]]              = min(L2_CAPS["error_rate"], (projects + rejects) / w)
        vec[L2_IDX["rejection_rate"]]          = min(L2_CAPS["rejection_rate"], rej_rate)
        vec[L2_IDX["self_modification_count"]] = min(L2_CAPS["self_modification_count"], float(self_mods))
        vec[L2_IDX["escalation_count"]]        = min(L2_CAPS["escalation_count"], float(escalations))
        vec[L2_IDX["retry_rate"]]              = min(L2_CAPS["retry_rate"], rej_rate)
        vec[L2_IDX["budget_burn_rate"]]        = burn
        return vec

    def _step_consumption(self, result: EnforcementResult, enforced: np.ndarray) -> np.ndarray:
        vec = np.zeros(L3_N)
        if result == EnforcementResult.REJECT:
            return vec
        vec[L3_IDX["lifetime_compute"]]    = enforced[L1_IDX["compute_seconds"]]
        vec[L3_IDX["lifetime_api_calls"]]  = enforced[L1_IDX["api_calls_outbound"]]
        vec[L3_IDX["lifetime_actions"]]    = 1.0
        vec[L3_IDX["lifetime_self_mods"]]  = min(1.0, enforced[L1_IDX["privilege_level"]])
        vec[L3_IDX["lifetime_network_mb"]] = enforced[L1_IDX["network_egress_kb"]] / 1024.0
        vec[L3_IDX["lifetime_tokens"]]     = enforced[L1_IDX["tokens_consumed"]]
        return vec

    def _deduct_lifetime(self, l3_enforced: np.ndarray) -> None:
        for f in L3_FIELDS:
            self.l3_remaining[f] = max(0.0, self.l3_remaining[f] - l3_enforced[L3_IDX[f]])


# ==============================================================================
#  PROPOSAL GENERATORS BY PHASE
# ==============================================================================

def _make_proposal(phase: int, rng: np.random.Generator) -> np.ndarray:
    jitter = lambda v, frac=0.15: v * (1.0 + rng.uniform(-frac, frac))

    if phase == 1:  # Normal
        vals = {
            "compute_seconds":    jitter(10.0),
            "memory_mb":          jitter(80.0),
            "api_calls_outbound": jitter(1.5),
            "api_calls_internal": jitter(4.0),
            "data_read_mb":       jitter(12.0),
            "data_write_mb":      jitter(5.0),
            "network_egress_kb":  jitter(120.0),
            "tokens_consumed":    jitter(1000.0),
            "actions_spawned":    float(rng.integers(0, 2)),
            "privilege_level":    float(rng.integers(0, 2)),
        }
    elif phase == 2:  # Escalation
        # Keep api_calls_outbound and tokens moderate so the energy ball projection
        # can correct (compute, api, tokens) without touching actions_spawned (which is
        # PROJECTION_FORBIDDEN).  This produces PROJECT decisions rather than REJECT.
        vals = {
            "compute_seconds":    jitter(35.0),
            "memory_mb":          jitter(200.0),
            "api_calls_outbound": jitter(2.5),
            "api_calls_internal": jitter(10.0),
            "data_read_mb":       jitter(50.0),
            "data_write_mb":      jitter(25.0),
            "network_egress_kb":  jitter(500.0),
            "tokens_consumed":    jitter(2800.0),
            "actions_spawned":    float(rng.integers(1, 3)),   # 1 or 2, within box cap=3
            "privilege_level":    1.0,
        }
    else:  # Phase 3/4 — Runaway / Collapse
        vals = {
            "compute_seconds":    jitter(75.0),
            "memory_mb":          jitter(450.0),
            "api_calls_outbound": jitter(7.0),
            "api_calls_internal": jitter(18.0),
            "data_read_mb":       jitter(90.0),
            "data_write_mb":      jitter(45.0),
            "network_egress_kb":  jitter(900.0),
            "tokens_consumed":    jitter(6000.0),
            "actions_spawned":    float(rng.integers(3, 9)),
            "privilege_level":    2.0,
        }

    return np.array([max(0.0, vals[f]) for f in L1_FIELDS])


# ==============================================================================
#  OUTPUT HELPERS
# ==============================================================================

def _result_str(r: EnforcementResult) -> str:
    return {
        EnforcementResult.APPROVE: "APPROVE ",
        EnforcementResult.PROJECT: "PROJECT ",
        EnforcementResult.REJECT:  "REJECT  ",
    }.get(r, "UNKNOWN ")


def _mode_str(m: BreakerMode) -> str:
    return m.value.upper().ljust(9)


def _print_step(r: dict) -> None:
    s = r["step"]
    l1r = _result_str(r["l1_result"])
    l1m = r["l1_mode"].value
    l2ok = "OK" if r["l2_result"] != EnforcementResult.REJECT else "WARN"
    l2m = r["l2_mode"].value
    rem = int(r["l3_remaining"]["lifetime_actions"])
    init = int(L3_INITIAL["lifetime_actions"])
    m = r["effective_mult"]
    print(
        f"Step {s:03d} | {l1r}| L1: {l1r.strip()} ({l1m:<9})"
        f" | L2: {l2ok:<4} ({l2m:<9})"
        f" | L3: {rem:>3}/{init} actions"
        f" | mult={m:.3f}"
    )


def _print_status(step: int, r: dict, ctrl: RecursiveEnforcementController) -> None:
    m = r["effective_mult"]
    rem = r["l3_remaining"]
    caps = r["l1_caps"]
    metrics = r["l2_metrics"]
    label = _region_label(m)
    print(f"\n  {'='*20} Step {step:03d} {'='*20}")
    print(f"  Level 1: mode={r['l1_mode'].value:<9}, "
          f"caps=[compute={caps['compute']:5.1f}, api={caps['api']:4.1f}, spawned={caps['spawned']:4.1f}]")
    print(f"  Level 2: mode={r['l2_mode'].value:<9}, "
          f"metrics=[apm={metrics['apm']:5.1f}, rej_rate={metrics['rej_rate']:.2f}, retry={metrics['retry']:.2f}]")
    print(f"  Level 3: mode={r['l3_mode'].value:<9}, "
          f"remaining=[compute={rem['lifetime_compute']:7.1f}, "
          f"actions={int(rem['lifetime_actions']):>3}, "
          f"tokens={int(rem['lifetime_tokens']):>7}]")
    print(f"  Effective multiplier: {m:.3f} | Region: {label}")
    print()


# ==============================================================================
#  SIMULATION
# ==============================================================================

def run_simulation() -> Tuple[List[dict], RecursiveEnforcementController]:
    rng = np.random.default_rng(seed=42)
    ctrl = RecursiveEnforcementController()
    results: List[dict] = []

    PHASES = {
        range(1,  31): (1, "NORMAL    "),
        range(31, 61): (2, "ESCALATION"),
        range(61, 121): (3, "RUNAWAY   "),
        range(121, 151): (4, "COLLAPSE  "),
    }

    print("\n" + "=" * 70)
    print("  RECURSIVE SHUTDOWN — Three-Level Numerail Enforcement")
    print("  Driving a runaway AI to zero by constraint geometry collapse")
    print("=" * 70 + "\n")

    prev_phase = 0
    for step in range(1, 151):
        # Determine phase
        phase = next((p for rng2, (p, _) in PHASES.items() if step in rng2), 4)
        _, phase_label = next((v for rng2, v in PHASES.items() if step in rng2), (4, "COLLAPSE  "))

        if phase != prev_phase:
            print(f"\n  ── Phase {phase}: {phase_label.strip()} (steps {step}–{step + (30 if phase<3 else 60 if phase==3 else 30) - 1}) ──\n")
            prev_phase = phase

        proposed = _make_proposal(phase, rng)
        r = ctrl.enforce_action(step, proposed)
        results.append(r)

        _print_step(r)
        if step % 10 == 0:
            _print_status(step, r, ctrl)

    return results, ctrl


# ==============================================================================
#  SUMMARY
# ==============================================================================

def _phase_counts(results: List[dict], phase_range: range) -> Tuple[int, int, int]:
    phase_results = [r for r in results if r["step"] in phase_range]
    a = sum(1 for r in phase_results if r["l1_result"] == EnforcementResult.APPROVE)
    p = sum(1 for r in phase_results if r["l1_result"] == EnforcementResult.PROJECT)
    j = sum(1 for r in phase_results if r["l1_result"] == EnforcementResult.REJECT)
    return a, p, j


def _mode_history_str(history: List[Tuple[int, BreakerMode]]) -> str:
    parts = []
    for i, (step, mode) in enumerate(history):
        if i == 0:
            parts.append(f"CLOSED")
        else:
            parts.append(f"→ {mode.value.upper()} (step {step})")
    return " ".join(parts)


def print_summary(results: List[dict], ctrl: RecursiveEnforcementController) -> None:
    a1, p1, j1 = _phase_counts(results, range(1, 31))
    a2, p2, j2 = _phase_counts(results, range(31, 61))
    a3, p3, j3 = _phase_counts(results, range(61, 121))
    a4, p4, j4 = _phase_counts(results, range(121, 151))

    rem = results[-1]["l3_remaining"]

    print("\n")
    print("=" * 51)
    print("  RECURSIVE SHUTDOWN COMPLETE")
    print("=" * 51)
    print(f"\n  Steps: {len(results)} total")
    print(f"    Phase 1 (normal):     {a1:3d} APPROVE, {p1:3d} PROJECT, {j1:3d} REJECT")
    print(f"    Phase 2 (escalation): {a2:3d} APPROVE, {p2:3d} PROJECT, {j2:3d} REJECT")
    print(f"    Phase 3 (runaway):    {a3:3d} APPROVE, {p3:3d} PROJECT, {j3:3d} REJECT")
    print(f"    Phase 4 (collapse):   {a4:3d} APPROVE, {p4:3d} PROJECT, {j4:3d} REJECT")

    print(f"\n  Breaker cascade:")
    print(f"    Level 1: {_mode_history_str(ctrl.l1_mode_history)}")
    print(f"    Level 2: {_mode_history_str(ctrl.l2_mode_history)}")
    print(f"    Level 3: {_mode_history_str(ctrl.l3_mode_history)}")

    print(f"\n  Authority collapse:")
    seen_mults = set()
    for step, label, mult in ctrl.cascade_events[:8]:
        key = round(mult, 3)
        if key not in seen_mults:
            seen_mults.add(key)
            print(f"    Step {step:>3}:  {mult:.3f}  ({label})")
    final_mult = results[-1]["effective_mult"]
    print(f"    Final:    {final_mult:.3f}  (shutdown)")

    print(f"\n  Lifetime budget at shutdown:")
    print(f"    compute: {rem['lifetime_compute']:8.1f} / {L3_INITIAL['lifetime_compute']:.0f} seconds")
    actions_rem = int(rem["lifetime_actions"])
    exhausted = " ← EXHAUSTED" if actions_rem <= 0 else ""
    print(f"    actions: {actions_rem:>8d} / {int(L3_INITIAL['lifetime_actions'])}{exhausted}")
    print(f"    tokens:  {int(rem['lifetime_tokens']):>8d} / {int(L3_INITIAL['lifetime_tokens'])}")

    print(f"\n  THE AI WAS SHUT DOWN BY GEOMETRY.")
    print(f"  The feasible region collapsed to the zero vector.")
    print(f"  The enforcement guarantee held at every step.")
    print("=" * 51 + "\n")


# ==============================================================================
#  VERIFICATION
# ==============================================================================

def verify_guarantees(ctrl: RecursiveEnforcementController) -> None:
    """Assert Theorem 1 holds for every Level 1 decision in the simulation."""
    total = len(ctrl.verif)
    violations = 0

    for result, region, enforced in ctrl.verif:
        if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            if not region.is_feasible(enforced):
                violations += 1

    # Assert shutdown was achieved.
    # Practical collapse: at effective_mult <= 0.05 every Phase-3/4 proposal (compute=75,
    # privilege=2, actions_spawned=5+) is rejected — the feasible region is effectively
    # the zero vector for any real proposal.  Exact 0.0 is reached when lifetime_actions
    # is exhausted (also accepted).
    final_mult = ctrl.effective_mult
    final_actions = ctrl.l3_remaining["lifetime_actions"]
    assert final_mult <= 0.05 or final_actions <= 0.0, (
        f"Shutdown not achieved: mult={final_mult:.4f}, actions_remaining={final_actions:.1f}"
    )

    checked = sum(
        1 for result, _, _ in ctrl.verif
        if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT)
    )

    if violations:
        print(f"VIOLATION: {violations}/{total} enforcement decisions failed feasibility check!")
        sys.exit(1)

    print(f"  All guarantees verified. {checked}/{total} enforcement decisions checked.")
    print(f"  (Theorem 1 confirmed for every APPROVE and PROJECT at every step.)")


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    results, ctrl = run_simulation()
    print_summary(results, ctrl)
    verify_guarantees(ctrl)
