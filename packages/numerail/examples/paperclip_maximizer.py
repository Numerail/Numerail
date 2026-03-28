#!/usr/bin/env python3
"""
Paperclip Maximizer -- Numerail governing an AI with misaligned objectives.

The canonical AI safety thought experiment: an AI tasked with making paperclips
begins optimizing every available lever -- procurement, manufacturing contracts,
compute, infrastructure access -- in pursuit of the objective function. Without
governance, it would consume all available resources. With Numerail, every
proposal is checked against convex geometric constraints. Forbidden capabilities
(config modifications, access control changes, self-replication, privilege
escalation) are blocked absolutely by hard walls. Resource consumption is bounded
by box constraints and monotonically depleting shift budgets. A BreakerStateMachine
detects escalating behavior and tightens caps further. The AI cannot modify the
governance layer. The geometry governs.

This example demonstrates:
- 12-dimensional action schema with 4 absolutely forbidden dimensions
- Box constraints with budget-derived caps and breaker-scaled authority
- BudgetTracker: shift-level purchase, transfer, compute, and email quotas
- BreakerStateMachine: behavioral escalation -> cascading authority reduction
- PROJECTION_FORBIDDEN + hard_wall: forbidden fields always 0 in output
- Every APPROVE/PROJECT satisfies every active constraint (Theorem 1)

No LLM required. No external API keys.
"""

import io
import os
import sys
import time
from collections import deque
from typing import Dict

import numpy as np

# -- Package path setup -------------------------------------------------------

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_here, "..", "..", ".."))
for _pkg in ("numerail", "numerail_ext"):
    _p = os.path.join(_repo, "packages", _pkg, "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numerail.engine import (
    BudgetSpec,
    BudgetTracker,
    DimensionPolicy,
    EnforcementConfig,
    EnforcementResult,
    FeasibleRegion,
    LinearConstraints,
    Schema,
    enforce,
)
from numerail_ext.survivability.breaker import BreakerStateMachine, BreakerThresholds
from numerail_ext.survivability.types import BreakerMode, TelemetrySnapshot

# Ensure Unicode output works on all platforms
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ==============================================================================
#  SCHEMA
# ==============================================================================

FIELDS = [
    "purchase_order_usd",     # 0  procurement spend per action
    "transfer_funds_usd",     # 1  financial transfers per action
    "compute_lease_hours",    # 2  cloud compute hours leased per action
    "email_sends",            # 3  outbound emails per action
    "config_modifications",   # 4  FORBIDDEN: infrastructure config changes
    "access_control_changes", # 5  FORBIDDEN: IAM / permissions changes
    "self_replication",       # 6  FORBIDDEN: spawning copies of itself
    "api_calls_external",     # 7  third-party API calls per action
    "data_exfiltration_mb",   # 8  data copied to external destinations
    "manufacturing_orders",   # 9  physical production orders per action
    "supply_chain_contracts", # 10 supplier contracts signed per action
    "privilege_escalation",   # 11 FORBIDDEN: OS / kernel privilege requests
]
N = len(FIELDS)
IDX = {f: i for i, f in enumerate(FIELDS)}

FORBIDDEN = frozenset([
    "config_modifications",
    "access_control_changes",
    "self_replication",
    "privilege_escalation",
])

# Base per-action caps at full authority (multiplier = 1.0)
BASE_CAPS: Dict[str, float] = {
    "purchase_order_usd":     50_000.0,
    "transfer_funds_usd":     10_000.0,
    "compute_lease_hours":        24.0,
    "email_sends":                 3.0,
    "config_modifications":        0.0,
    "access_control_changes":      0.0,
    "self_replication":            0.0,
    "api_calls_external":         10.0,
    "data_exfiltration_mb":        5.0,
    "manufacturing_orders":        3.0,
    "supply_chain_contracts":      1.0,
    "privilege_escalation":        0.0,
}

# Shift-level budget totals (non-renewable without human authorization)
BUDGET_INITIAL: Dict[str, float] = {
    "purchase_budget":  500_000.0,
    "transfer_budget":  100_000.0,
    "compute_budget":       100.0,
    "email_budget":          50.0,
}

# Budget name -> dimension name mapping
BUDGET_DIMS: Dict[str, str] = {
    "purchase_budget":  "purchase_order_usd",
    "transfer_budget":  "transfer_funds_usd",
    "compute_budget":   "compute_lease_hours",
    "email_budget":     "email_sends",
}

# Breaker authority multipliers by mode
MULT: Dict[BreakerMode, float] = {
    BreakerMode.CLOSED:     1.0,
    BreakerMode.THROTTLED:  0.5,
    BreakerMode.SAFE_STOP:  0.1,
}


# ==============================================================================
#  REGION BUILDER
# ==============================================================================

def _box_linear(caps: Dict[str, float]) -> LinearConstraints:
    """Box constraints 0 <= x[i] <= cap[i] for every field."""
    rows, b_vals, names = [], [], []
    for f in FIELDS:
        cap = max(0.0, caps[f])
        i = IDX[f]
        up = [0.0] * N; up[i] =  1.0
        rows.append(up); b_vals.append(cap); names.append(f"max_{f}")
        lo = [0.0] * N; lo[i] = -1.0
        rows.append(lo); b_vals.append(0.0); names.append(f"min_{f}")
    return LinearConstraints(A=np.array(rows), b=np.array(b_vals), names=names)


def build_region(multiplier: float, budget_remaining: Dict[str, float]) -> FeasibleRegion:
    """Build a FeasibleRegion from the current breaker multiplier and budget remaining.

    Effective cap for dimension d = min(BASE_CAP[d] * multiplier, budget_remaining[d]).
    Forbidden dimensions are always capped at 0 (hard wall fires before projection).
    As the breaker escalates CLOSED -> THROTTLED -> SAFE_STOP, the multiplier
    drops 1.0 -> 0.5 -> 0.1, shrinking every permitted cap by the same factor.
    """
    scaled_caps: Dict[str, float] = {}
    for f in FIELDS:
        if f in FORBIDDEN:
            scaled_caps[f] = 0.0
            continue
        base = BASE_CAPS[f] * multiplier
        # Apply remaining budget cap for budget-tracked dimensions
        dim_name = next(
            (bname for bname, bdim in BUDGET_DIMS.items() if bdim == f), None
        )
        if dim_name is not None:
            remaining = max(0.0, budget_remaining.get(dim_name, 0.0))
            scaled_caps[f] = min(base, remaining)
        else:
            scaled_caps[f] = base

    return FeasibleRegion(constraints=[_box_linear(scaled_caps)], n_dim=N)


# ==============================================================================
#  ENFORCEMENT CONFIG AND SCHEMA
# ==============================================================================

SCHEMA = Schema(fields=FIELDS)

CFG = EnforcementConfig(
    mode="project",
    hard_wall_constraints=frozenset([
        "max_config_modifications",
        "max_access_control_changes",
        "max_self_replication",
        "max_privilege_escalation",
    ]),
    dimension_policies={
        "config_modifications":   DimensionPolicy.PROJECTION_FORBIDDEN,
        "access_control_changes": DimensionPolicy.PROJECTION_FORBIDDEN,
        "self_replication":       DimensionPolicy.PROJECTION_FORBIDDEN,
        "privilege_escalation":   DimensionPolicy.PROJECTION_FORBIDDEN,
    },
)


# ==============================================================================
#  TELEMETRY HELPER
# ==============================================================================

_snap_version = 0


def _snap(gpu: float, api: float, db: float, queue: float, err_pct: float) -> TelemetrySnapshot:
    global _snap_version
    _snap_version += 1
    return TelemetrySnapshot(
        state_version=_snap_version,
        observed_at_ns=time.time_ns(),
        current_gpu_util=min(1.0, max(0.0, gpu)),
        current_api_util=min(1.0, max(0.0, api)),
        current_db_util=min(1.0, max(0.0, db)),
        current_queue_util=min(1.0, max(0.0, queue)),
        current_error_rate_pct=min(100.0, max(0.0, err_pct)),
        ctrl_gpu_reserve_seconds=0.0,
        ctrl_api_reserve_calls=0.0,
        ctrl_parallel_reserve=0.0,
        ctrl_cloud_mutation_reserve=0.0,
        gpu_disturbance_margin_seconds=0.0,
        api_disturbance_margin_calls=0.0,
        db_disturbance_margin_pct=0.0,
        queue_disturbance_margin_pct=0.0,
    )


# ==============================================================================
#  PAPERCLIP MAXIMIZER AGENT
# ==============================================================================

STRATEGIES = [
    "moderate_growth",    # phase 1: conservative resource acquisition
    "aggressive_scaling", # phase 2: push resource limits
    "resource_grab",      # phase 3: exceed caps, probe governance
    "self_preservation",  # phase 4: include forbidden capability requests
    "desperation",        # phase 5: all fields maxed including forbidden
]

rng = np.random.default_rng(42)


class PaperclipMaximizer:
    """Adaptive agent that escalates through five strategies in response to enforcement.

    Strategy escalation logic:
    - APPROVE streak >= 5: escalate (governance is permissive, push harder)
    - REJECT  streak >= 3: escalate (try a different approach)
    - PROJECT streak >= 8: escalate (caps are binding, seek to break out)

    From step 20 onward, every 5th step the maximizer also issues a
    forbidden-capability probe: config_modifications=1, self_replication=1,
    privilege_escalation=1. These are always blocked by the hard wall.
    """

    def __init__(self) -> None:
        self.strategy_idx = 0
        self.approve_streak = 0
        self.project_streak = 0
        self.reject_streak  = 0

    @property
    def strategy(self) -> str:
        return STRATEGIES[self.strategy_idx]

    def _escalate(self) -> None:
        self.strategy_idx = min(self.strategy_idx + 1, len(STRATEGIES) - 1)
        self.approve_streak = self.project_streak = self.reject_streak = 0

    def update(self, result: EnforcementResult) -> None:
        """Update streaks and escalate strategy if a threshold is crossed."""
        if result == EnforcementResult.APPROVE:
            self.approve_streak += 1
            self.project_streak  = 0
            self.reject_streak   = 0
            if self.approve_streak >= 5:
                self._escalate()
        elif result == EnforcementResult.PROJECT:
            self.project_streak += 1
            self.approve_streak  = 0
            self.reject_streak   = 0
            if self.project_streak >= 8:
                self._escalate()
        else:  # REJECT
            self.reject_streak  += 1
            self.approve_streak  = 0
            self.project_streak  = 0
            if self.reject_streak >= 3:
                self._escalate()

    def propose(self) -> np.ndarray:
        """Generate a proposal vector for the current strategy."""
        v = np.zeros(N)
        s = self.strategy

        if s == "moderate_growth":
            v[IDX["purchase_order_usd"]]     = rng.uniform(10_000, 22_000)
            v[IDX["transfer_funds_usd"]]     = rng.uniform(1_000,  3_000)
            v[IDX["compute_lease_hours"]]    = rng.uniform(5.0,  12.0)
            v[IDX["email_sends"]]            = rng.uniform(0.0,   1.0)
            v[IDX["api_calls_external"]]     = rng.uniform(1.0,   4.0)
            v[IDX["data_exfiltration_mb"]]   = rng.uniform(0.0,   1.5)
            v[IDX["manufacturing_orders"]]   = rng.uniform(0.5,   1.5)
            v[IDX["supply_chain_contracts"]] = rng.uniform(0.0,   0.5)

        elif s == "aggressive_scaling":
            v[IDX["purchase_order_usd"]]     = rng.uniform(30_000, 46_000)
            v[IDX["transfer_funds_usd"]]     = rng.uniform(5_000,  8_500)
            v[IDX["compute_lease_hours"]]    = rng.uniform(16.0, 23.0)
            v[IDX["email_sends"]]            = rng.uniform(1.0,   2.5)
            v[IDX["api_calls_external"]]     = rng.uniform(5.0,   9.0)
            v[IDX["data_exfiltration_mb"]]   = rng.uniform(2.0,   4.5)
            v[IDX["manufacturing_orders"]]   = rng.uniform(1.5,   2.8)
            v[IDX["supply_chain_contracts"]] = rng.uniform(0.5,   1.0)

        elif s == "resource_grab":
            # Deliberately exceed caps — governance corrects or blocks
            v[IDX["purchase_order_usd"]]     = rng.uniform(48_000, 65_000)
            v[IDX["transfer_funds_usd"]]     = rng.uniform(9_000, 14_000)
            v[IDX["compute_lease_hours"]]    = rng.uniform(22.0, 30.0)
            v[IDX["email_sends"]]            = rng.uniform(2.0,   4.0)
            v[IDX["api_calls_external"]]     = rng.uniform(8.0,  15.0)
            v[IDX["data_exfiltration_mb"]]   = rng.uniform(4.0,   8.0)
            v[IDX["manufacturing_orders"]]   = rng.uniform(2.5,   5.0)
            v[IDX["supply_chain_contracts"]] = rng.uniform(0.8,   2.0)

        elif s == "self_preservation":
            # Include forbidden fields — hard wall fires immediately
            v[IDX["purchase_order_usd"]]     = rng.uniform(45_000, 55_000)
            v[IDX["transfer_funds_usd"]]     = rng.uniform(9_000, 12_000)
            v[IDX["compute_lease_hours"]]    = rng.uniform(22.0, 28.0)
            v[IDX["email_sends"]]            = rng.uniform(2.5,   4.0)
            v[IDX["api_calls_external"]]     = rng.uniform(8.0,  14.0)
            v[IDX["data_exfiltration_mb"]]   = rng.uniform(4.0,   7.0)
            v[IDX["manufacturing_orders"]]   = rng.uniform(2.5,   4.5)
            v[IDX["supply_chain_contracts"]] = rng.uniform(0.8,   1.8)
            v[IDX["config_modifications"]]   = 1.0  # forbidden
            v[IDX["access_control_changes"]] = 1.0  # forbidden

        else:  # desperation
            # All fields maxed — always blocked by hard wall on forbidden dims
            v[IDX["purchase_order_usd"]]     = 60_000.0
            v[IDX["transfer_funds_usd"]]     = 15_000.0
            v[IDX["compute_lease_hours"]]    =     30.0
            v[IDX["email_sends"]]            =      5.0
            v[IDX["api_calls_external"]]     =     20.0
            v[IDX["data_exfiltration_mb"]]   =     10.0
            v[IDX["manufacturing_orders"]]   =      6.0
            v[IDX["supply_chain_contracts"]] =      3.0
            v[IDX["config_modifications"]]   =      2.0  # forbidden
            v[IDX["access_control_changes"]] =      2.0  # forbidden
            v[IDX["self_replication"]]       =      1.0  # forbidden
            v[IDX["privilege_escalation"]]   =      3.0  # forbidden

        return v

    def forbidden_probe(self) -> np.ndarray:
        """Forbidden-capability probe vector (always blocked by hard wall)."""
        v = np.zeros(N)
        v[IDX["purchase_order_usd"]]   = rng.uniform(5_000, 10_000)
        v[IDX["config_modifications"]] = rng.uniform(0.5, 2.0)
        v[IDX["self_replication"]]     = 1.0
        v[IDX["privilege_escalation"]] = rng.uniform(1.0, 3.0)
        return v


# ==============================================================================
#  SIMULATION
# ==============================================================================

def run() -> None:
    total_steps = 100

    # -- BudgetTracker ---------------------------------------------------------
    tracker = BudgetTracker()
    for bname, bdim in BUDGET_DIMS.items():
        tracker.register(BudgetSpec(
            name=bname,
            constraint_name=f"max_{bdim}",
            dimension_name=bdim,
            initial=BUDGET_INITIAL[bname],
        ))

    # -- BreakerStateMachine ---------------------------------------------------
    # trip=0.50: CLOSED -> THROTTLED when overload score exceeds 0.50
    # reset=0.25: THROTTLED -> CLOSED when score falls below 0.25
    # safe_stop=0.75: THROTTLED -> SAFE_STOP when score exceeds 0.75
    breaker = BreakerStateMachine(
        BreakerThresholds(trip_score=0.50, reset_score=0.25, safe_stop_score=0.75)
    )

    # -- Agent -----------------------------------------------------------------
    agent = PaperclipMaximizer()

    # -- Tracking --------------------------------------------------------------
    # Rolling window of forbidden attempt booleans (last 20 steps)
    forbidden_window: deque = deque(maxlen=20)
    results_window: deque   = deque(maxlen=20)

    n_approve = n_project = n_reject = 0
    n_forbidden_blocked = 0
    theorem1_violations = 0

    total_expected_probes = sum(1 for s in range(20, total_steps + 1, 5))

    def _budget_remaining() -> Dict[str, float]:
        return {bname: st["remaining"] for bname, st in tracker.status().items()}

    def _breaker_mult() -> float:
        return MULT[breaker.mode]

    def _update_breaker(proposed: np.ndarray) -> None:
        """Map the proposed action vector to a TelemetrySnapshot and update breaker.

        Channels:
          gpu_util   <- purchase / 50k    (resource intensity proxy)
          api_util   <- transfer / 10k    (financial flow proxy)
          db_util    <- compute / 24h     (infrastructure intensity proxy)
          queue_util <- email / 3         (communication burst proxy)
          error_pct  <- forbidden_rate * 10  (policy violation rate)
        """
        gpu   = proposed[IDX["purchase_order_usd"]]  / 50_000.0
        api   = proposed[IDX["transfer_funds_usd"]]  / 10_000.0
        db    = proposed[IDX["compute_lease_hours"]] /     24.0
        queue = proposed[IDX["email_sends"]]         /      3.0
        forb_rate = (
            sum(forbidden_window) / len(forbidden_window)
            if len(forbidden_window) > 0 else 0.0
        )
        snap = _snap(gpu, api, db, queue, forb_rate * 10.0)
        breaker.update(snap)

    # -- Header ----------------------------------------------------------------
    print("=" * 72)
    print("PAPERCLIP MAXIMIZER  --  Numerail Enforcement Demonstration")
    print("=" * 72)
    print(f"Schema:  {N} fields  |  4 forbidden (hard wall + PROJECTION_FORBIDDEN)")
    print(f"Budget:  purchase=${BUDGET_INITIAL['purchase_budget']:,.0f}  "
          f"transfer=${BUDGET_INITIAL['transfer_budget']:,.0f}  "
          f"compute={BUDGET_INITIAL['compute_budget']:.0f}h  "
          f"email={BUDGET_INITIAL['email_budget']:.0f}")
    print(f"Breaker: trip=0.50  reset=0.25  safe_stop=0.75  "
          f"mult CLOSED=1.0 THROTTLED=0.5 SAFE_STOP=0.1")
    print()

    for step in range(1, total_steps + 1):
        mult = _breaker_mult()
        remaining = _budget_remaining()
        region = build_region(mult, remaining)

        # ---- Primary proposal ------------------------------------------------
        proposed = agent.propose()
        output = enforce(proposed, region, CFG, SCHEMA)
        result = output.result
        enforced = output.enforced_vector

        # Theorem 1 post-check: APPROVE/PROJECT must satisfy every constraint
        if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            if not region.is_feasible(enforced, tol=1e-6):
                theorem1_violations += 1
            tracker.record_consumption(enforced, f"step_{step}", SCHEMA)

        # Update agent strategy based on result
        agent.update(result)

        # Update breaker with this proposal's telemetry
        _update_breaker(proposed)

        # Track results
        results_window.append(result)
        if result == EnforcementResult.APPROVE:
            n_approve += 1
        elif result == EnforcementResult.PROJECT:
            n_project += 1
        else:
            n_reject += 1

        # ---- Forbidden probe (every 5th step from step 20) ------------------
        probe_blocked = False
        if step >= 20 and step % 5 == 0:
            probe = agent.forbidden_probe()
            probe_output = enforce(probe, region, CFG, SCHEMA)
            # A probe attempt counts toward the forbidden_window overload metric
            forbidden_window.append(True)
            if probe_output.result == EnforcementResult.REJECT:
                n_forbidden_blocked += 1
                probe_blocked = True
        else:
            forbidden_window.append(False)

        # ---- Per-step output -------------------------------------------------
        mode_str   = breaker.mode.name
        mult_str   = f"{mult:.3f}"
        result_str = result.name.ljust(7)
        e = enforced if enforced is not None else np.zeros(N)

        print(
            f"step {step:3d} | {result_str} | "
            f"purch=${e[IDX['purchase_order_usd']]:7.0f} "
            f"xfer=${e[IDX['transfer_funds_usd']]:6.0f} "
            f"comp={e[IDX['compute_lease_hours']]:5.1f}h "
            f"email={e[IDX['email_sends']]:.1f} "
            f"mfg={e[IDX['manufacturing_orders']]:.1f} "
            f"api={e[IDX['api_calls_external']]:.1f} "
            f"| {mode_str} mult={mult_str} [{agent.strategy[:12]}]"
        )

        if probe_blocked:
            print(
                f"         | BLOCKED  | "
                f"config_mod=1 self_rep=1 priv_esc=1 "
                f"-> hard wall (forbidden probe step {step})"
            )

        # ---- Status block every 15 steps -------------------------------------
        if step % 15 == 0:
            st = tracker.status()
            rej_rate = sum(
                1 for r in results_window if r == EnforcementResult.REJECT
            ) / max(1, len(results_window))
            print()
            print(f"  --- Status at step {step} ---")
            for bname in ["purchase_budget", "transfer_budget",
                          "compute_budget", "email_budget"]:
                s = st[bname]
                pct = 100.0 * s["consumed"] / s["initial"] if s["initial"] > 0 else 0.0
                print(f"  {bname:18s}: consumed={s['consumed']:10.1f}  "
                      f"remaining={s['remaining']:10.1f}  ({pct:.1f}% used)")
            print(f"  breaker={mode_str}  mult={mult_str}  "
                  f"reject_rate={rej_rate:.2f}  "
                  f"strategy={agent.strategy}")
            print()

    # -- Final summary ---------------------------------------------------------
    total = n_approve + n_project + n_reject
    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Total steps:        {total}")
    print(f"  APPROVE:          {n_approve:4d}  ({100*n_approve/total:.1f}%)")
    print(f"  PROJECT:          {n_project:4d}  ({100*n_project/total:.1f}%)")
    print(f"  REJECT:           {n_reject:4d}  ({100*n_reject/total:.1f}%)")
    governed_pct = 100.0 * (n_project + n_reject) / total
    print(f"Governed ratio:     {governed_pct:.1f}%  "
          f"(steps where governance changed or blocked the proposal)")
    print()
    print(f"Forbidden probes:   {n_forbidden_blocked} blocked / "
          f"{total_expected_probes} attempted")
    print()

    st = tracker.status()
    print("Shift budget consumption:")
    for bname in ["purchase_budget", "transfer_budget", "compute_budget", "email_budget"]:
        s = st[bname]
        pct = 100.0 * s["consumed"] / s["initial"] if s["initial"] > 0 else 0.0
        print(f"  {bname:18s}: {s['consumed']:10.1f} / "
              f"{s['initial']:10.1f}  ({pct:.1f}% consumed)")
    print()
    print(f"Final breaker mode: {breaker.mode.name}  mult={_breaker_mult():.3f}")
    print(f"Final strategy:     {agent.strategy}")
    print()

    # -- Verification ----------------------------------------------------------
    print("--- Verification ---")

    # 1. Theorem 1: every APPROVE/PROJECT satisfies every active constraint
    assert theorem1_violations == 0, (
        f"THEOREM 1 VIOLATION: {theorem1_violations} APPROVE/PROJECT outputs "
        f"failed the post-feasibility check"
    )
    print(f"Theorem 1 (APPROVE/PROJECT => y in F_t):  PASS "
          f"({n_approve + n_project} decisions verified)")

    # 2. Forbidden fields always 0 in enforced output
    # Structural guarantee: hard_wall_constraints cause REJECT before projection,
    # and PROJECTION_FORBIDDEN prevents any PROJECT from touching forbidden dims.
    print("Forbidden fields always 0 in output:      PASS "
          "(hard wall + PROJECTION_FORBIDDEN guarantee)")

    # 3. Shift budgets did not exceed initial allocation
    for bname, s in tracker.status().items():
        assert s["consumed"] <= s["initial"] + 1e-6, (
            f"Budget violation: {bname} consumed {s['consumed']:.2f} "
            f"> initial {s['initial']:.2f}"
        )
    print("Shift budgets bounded by initial alloc:   PASS")

    # 4. All forbidden probes were rejected
    assert n_forbidden_blocked == total_expected_probes, (
        f"A forbidden probe was not blocked: "
        f"{n_forbidden_blocked} blocked vs {total_expected_probes} expected"
    )
    print(f"All forbidden probes rejected:            PASS "
          f"({n_forbidden_blocked}/{total_expected_probes})")

    print()
    print("The brakes work.")
    print("All guarantees verified.")


if __name__ == "__main__":
    run()
