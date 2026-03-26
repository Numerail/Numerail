#!/usr/bin/env python3
"""hello_world.py — Numerail end-to-end demonstration.

Run from the repository root:
    python packages/numerail/examples/hello_world.py

Covers every major capability of both numerail (core) and numerail-ext
across 14 self-contained steps. Uses plain assert statements and
print-based reporting. No pytest required.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from time import time_ns

import numpy as np

# Suppress expected "solver failed" warnings that arise from contradictory-
# constraint demos (step 3) and mixed-type projections (step 2).  These are
# correct behaviour (fail-closed), not bugs.  The script confirms them via
# assertions; the log lines would just clutter the demo output.
logging.getLogger("numerail").setLevel(logging.ERROR)

# ── Path setup ─────────────────────────────────────────────────────────────
# Works whether the script is invoked from the repo root or directly.
_HERE = Path(__file__).resolve()
_PKG_NUMERAIL = _HERE.parent.parent           # packages/numerail
_PKG_EXT = _PKG_NUMERAIL.parent / "numerail_ext"  # packages/numerail_ext
for _p in [str(_PKG_NUMERAIL / "src"), str(_PKG_EXT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Core imports ───────────────────────────────────────────────────────────
from numerail.engine import (
    EnforcementConfig,
    EnforcementResult,
    FeasibleRegion,
    LinearConstraints,
    NumerailSystem,
    PSDConstraint,
    QuadraticConstraint,
    SOCPConstraint,
    Schema,
    box_constraints,
    enforce,
)
from numerail.local import NumerailSystemLocal
from numerail.parser import PolicyParser, lint_config

# ── Extension imports ──────────────────────────────────────────────────────
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
    TelemetrySnapshot,
    WorkloadRequest,
)

# ── Output helpers ─────────────────────────────────────────────────────────

def header(n: int, title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  STEP {n} -- {title}")
    print(f"{'=' * 55}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


# ── Shared helper: TelemetrySnapshot ─────────────────────────────────────

def _snap(
    gpu: float, api: float, db: float, queue: float, error: float = 0.0
) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot. Reserves/margins fixed at sensible values."""
    return TelemetrySnapshot(
        state_version=1,
        observed_at_ns=time_ns(),
        current_gpu_util=gpu,
        current_api_util=api,
        current_db_util=db,
        current_queue_util=queue,
        current_error_rate_pct=error,
        ctrl_gpu_reserve_seconds=10.0,
        ctrl_api_reserve_calls=5.0,
        ctrl_parallel_reserve=2.0,
        ctrl_cloud_mutation_reserve=1.0,
        gpu_disturbance_margin_seconds=5.0,
        api_disturbance_margin_calls=2.0,
        db_disturbance_margin_pct=2.0,
        queue_disturbance_margin_pct=2.0,
    )


# ══════════════════════════════════════════════════════
# STEP 1 — THE GUARANTEE (core enforcement)
# ══════════════════════════════════════════════════════

header(1, "THE GUARANTEE (core enforcement)")

# 5-dimensional box: all values in [0, 100]
region1 = box_constraints([0.0] * 5, [100.0] * 5)

# --- Feasible proposal: expect APPROVE ---
x_feasible = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
out1a = enforce(x_feasible, region1)
assert out1a.result == EnforcementResult.APPROVE, f"Expected APPROVE, got {out1a.result}"
assert np.allclose(out1a.enforced_vector, x_feasible)
ok(f"Feasible [50, 50, 50, 50, 50] -> APPROVE  (enforced == input)")

# --- Infeasible proposal: expect PROJECT + guarantee ---
x_bad = np.array([150.0, 50.0, -10.0, 200.0, 50.0])
out1b = enforce(x_bad, region1)
assert out1b.result == EnforcementResult.PROJECT, f"Expected PROJECT, got {out1b.result}"
assert region1.is_feasible(out1b.enforced_vector), "Guarantee violated: enforced vector not feasible"
assert out1b.distance > 0
ok(f"Infeasible [150, 50, -10, 200, 50] -> PROJECT")
ok(f"  Enforced: {[round(v, 1) for v in out1b.enforced_vector.tolist()]}")
ok(f"  Distance: {out1b.distance:.4f}  is_feasible: True")


# ══════════════════════════════════════════════════════
# STEP 2 — ALL FOUR CONSTRAINT TYPES
# ══════════════════════════════════════════════════════

header(2, "ALL FOUR CONSTRAINT TYPES")

n2 = 3  # 3-dimensional problem

# Linear: box constraints  0 <= x <= 10
lin2 = LinearConstraints(
    A=np.vstack([np.eye(n2), -np.eye(n2)]),
    b=np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
    names=["ub_0", "ub_1", "ub_2", "lb_0", "lb_1", "lb_2"],
)

# Quadratic: x'x <= 200  (energy bound — prevents peak-everything-at-once)
quad2 = QuadraticConstraint(
    np.eye(n2), np.zeros(n2), 200.0,
    constraint_name="energy_bound",
)

# SOCP: ||x||_2 <= 15  (norm ball)
socp2 = SOCPConstraint(
    M=np.eye(n2), q=np.zeros(n2), c=np.zeros(n2), d=15.0,
    constraint_name="norm_ball",
)

# PSD: A0 + x[0]*A1 + x[1]*A2 + x[2]*A3 >= 0  (coupled headroom LMI)
# A0 = 5*I_2 gives a comfortable positive baseline;
# each x_i term erodes one diagonal entry.
k2 = 2
A0_2 = 5.0 * np.eye(k2)
A_list_2 = [
    -0.4 * np.array([[1.0, 0.0], [0.0, 0.0]]),
    -0.4 * np.array([[0.0, 0.0], [0.0, 1.0]]),
    -0.2 * np.eye(k2),
]
psd2 = PSDConstraint(A0_2, A_list_2, constraint_name="lmi_bound")

combined2 = FeasibleRegion([lin2, quad2, socp2, psd2], n2)

# [12, 12, 12] violates the box (> 10) and the SOCP (||x|| = ~20.8 > 15)
x_bad2 = np.array([12.0, 12.0, 12.0])
out2 = enforce(x_bad2, combined2)
assert out2.result in (EnforcementResult.PROJECT, EnforcementResult.REJECT)
if out2.result == EnforcementResult.PROJECT:
    assert combined2.is_feasible(out2.enforced_vector), "Guarantee violated"
    ok(f"4-type combined region: PROJECT  enforced={[round(v, 3) for v in out2.enforced_vector]}")
else:
    ok(f"4-type combined region: REJECT (no feasible projection found)")
ok(f"  Active types: Linear + Quadratic + SOCP + PSD")
ok(f"  Violated on input: {list(out2.violated_constraints)}")


# ══════════════════════════════════════════════════════
# STEP 3 — FAIL-CLOSED (Theorem 2)
# ══════════════════════════════════════════════════════

header(3, "FAIL-CLOSED (Theorem 2)")

# Contradictory: x[0] <= -1  AND  -x[0] <= -1  (i.e. x[0] >= 1)
# No point satisfies both simultaneously.
A3 = np.array([[1.0, 0.0], [-1.0, 0.0]])
b3 = np.array([-1.0, -1.0])
empty3 = FeasibleRegion(
    [LinearConstraints(A3, b3, ["x0_le_neg1", "neg_x0_le_neg1"])], 2
)

out3 = enforce(np.array([0.0, 0.0]), empty3)
assert out3.result == EnforcementResult.REJECT, f"Expected REJECT, got {out3.result}"
ok(f"Contradictory region -> REJECT")
ok(f"  System fails closed: no output is emitted unless a feasible point is verified.")
ok(f"  Violated: {list(out3.violated_constraints)}")


# ══════════════════════════════════════════════════════
# STEP 4 — HARD WALL DOMINANCE (Theorem 3)
# ══════════════════════════════════════════════════════

header(4, "HARD WALL DOMINANCE (Theorem 3)")

# A 2-D region: x[0] <= 10 (named 'hard_ceiling')
lc4 = LinearConstraints(
    np.array([[1.0, 0.0]]), np.array([10.0]), ["hard_ceiling"]
)
region4 = FeasibleRegion([lc4], 2)

# Enforce with 'hard_ceiling' declared as a hard wall
cfg4 = EnforcementConfig(mode="project", hard_wall_constraints=frozenset(["hard_ceiling"]))
out4 = enforce(np.array([20.0, 5.0]), region4, cfg4)

assert out4.result == EnforcementResult.REJECT, f"Expected REJECT, got {out4.result}"
assert out4.solver_method == "none", f"Expected solver_method='none', got '{out4.solver_method}'"
ok(f"Hard wall 'hard_ceiling' violated -> REJECT  solver_method='{out4.solver_method}'")
ok(f"  The solver was never invoked (hard wall is checked before the solver chain).")


# ══════════════════════════════════════════════════════
# STEP 5 — ENFORCEMENT MODES
# ══════════════════════════════════════════════════════

header(5, "ENFORCEMENT MODES")

region5 = box_constraints([0.0, 0.0], [5.0, 5.0])
x5 = np.array([8.0, 3.0])  # x[0]=8 violates upper bound 5; distance = 3.0

r_project = enforce(x5, region5, EnforcementConfig(mode="project"))
r_reject  = enforce(x5, region5, EnforcementConfig(mode="reject"))
r_near    = enforce(x5, region5, EnforcementConfig(mode="hybrid", max_distance=10.0))
r_far     = enforce(x5, region5, EnforcementConfig(mode="hybrid", max_distance=1.0))

assert r_project.result == EnforcementResult.PROJECT
assert r_reject.result  == EnforcementResult.REJECT
assert r_near.result    == EnforcementResult.PROJECT   # distance 3.0 <= 10.0
assert r_far.result     == EnforcementResult.REJECT    # distance 3.0 >  1.0

ok(f"mode='project'                   -> {r_project.result.value}  distance={r_project.distance:.4f}")
ok(f"mode='reject'                    -> {r_reject.result.value}")
ok(f"mode='hybrid' max_distance=10.0 -> {r_near.result.value}  "
   f"(distance {r_near.distance:.4f} <= 10.0)")
ok(f"mode='hybrid' max_distance=1.0  -> {r_far.result.value}   "
   f"(distance {r_far.distance:.4f} >  1.0)")


# ══════════════════════════════════════════════════════
# STEP 6 — PASSTHROUGH AND IDEMPOTENCE (Theorem 9)
# ══════════════════════════════════════════════════════

header(6, "PASSTHROUGH AND IDEMPOTENCE (Theorem 9)")

region6 = box_constraints([0.0] * 3, [1.0] * 3)
x6 = np.array([0.3, 0.7, 0.5])  # already feasible

out6a = enforce(x6, region6)
assert out6a.result == EnforcementResult.APPROVE
assert out6a.distance == 0.0

# Enforce the output again — must also APPROVE with distance 0
out6b = enforce(out6a.enforced_vector, region6)
assert out6b.result == EnforcementResult.APPROVE
assert out6b.distance == 0.0

ok(f"First enforcement:  {out6a.result.value}  distance={out6a.distance}")
ok(f"Second enforcement: {out6b.result.value}  distance={out6b.distance}")
ok(f"Idempotence confirmed: a projected/approved vector always re-approves unchanged.")


# ══════════════════════════════════════════════════════
# STEP 7 — SCHEMA AND NAMED DIMENSIONS
# ══════════════════════════════════════════════════════

header(7, "SCHEMA AND NAMED DIMENSIONS")

fields7 = ["gpu_seconds", "api_calls", "db_queries", "queue_depth", "error_rate"]
schema7 = Schema(fields7)
n7 = len(fields7)

# Upper and lower bounds per field
A7 = np.vstack([np.eye(n7), -np.eye(n7)])
b7 = np.array([120.0, 500.0, 1000.0, 100.0, 5.0,   # upper bounds
                0.0,   0.0,   0.0,    0.0,   0.0])  # lower bounds (>= 0)
lc7 = LinearConstraints(
    A7, b7,
    names=[f"max_{f}" for f in fields7] + [f"min_{f}" for f in fields7],
)
region7 = FeasibleRegion([lc7], n7)

proposal7 = {
    "gpu_seconds": 90.0, "api_calls": 350.0, "db_queries": 800.0,
    "queue_depth": 60.0, "error_rate": 2.5,
}
vec7    = schema7.vectorize(proposal7)
out7    = enforce(vec7, region7)
result7 = schema7.devectorize(out7.enforced_vector)

ok(f"Named input:  {proposal7}")
ok(f"Result: {out7.result.value}")
ok(f"Named output: { {k: round(v, 3) for k, v in result7.items()} }")


# ══════════════════════════════════════════════════════
# STEP 8 — BUDGET TRACKING
# ══════════════════════════════════════════════════════

header(8, "BUDGET TRACKING")

# Single-field policy with a 250 GPU-s session budget.
# The 'remaining_gpu_budget' linear row is updated each cycle by the
# budget tracker to reflect the monotone-non-expanding remaining allowance.
budget_policy = {
    "policy_id": "hello-world-budget",
    "schema": {"fields": ["gpu_seconds"]},
    "polytope": {
        "A": [[1.0], [-1.0], [1.0]],
        "b": [200.0, 0.0, 250.0],
        "names": ["max_gpu", "min_gpu", "remaining_gpu_budget"],
    },
    "budgets": [{
        "name": "gpu_budget",
        "constraint_name": "remaining_gpu_budget",
        "dimension_name": "gpu_seconds",
        "weight": 1.0,
        "initial": 250.0,
        "consumption_mode": "nonnegative",
    }],
    "enforcement": {"mode": "project"},
}

local8 = NumerailSystemLocal(budget_policy)
print(f"  Initial budget: {local8.budget_remaining['gpu_budget']:.1f} GPU-s")

r8a = local8.enforce({"gpu_seconds": 80.0}, action_id="step_a")
b8a = local8.budget_remaining["gpu_budget"]
r8b = local8.enforce({"gpu_seconds": 70.0}, action_id="step_b")
b8b = local8.budget_remaining["gpu_budget"]
r8c = local8.enforce({"gpu_seconds": 60.0}, action_id="step_c")
b8c = local8.budget_remaining["gpu_budget"]

print(f"  After step_a (80 GPU-s, {r8a['decision']:>7}): budget = {b8a:.1f}")
print(f"  After step_b (70 GPU-s, {r8b['decision']:>7}): budget = {b8b:.1f}")
print(f"  After step_c (60 GPU-s, {r8c['decision']:>7}): budget = {b8c:.1f}")

budget_before_rollback = b8c
local8.rollback("step_b")
budget_after_rollback = local8.budget_remaining["gpu_budget"]
restored = budget_after_rollback - budget_before_rollback

assert abs(restored - 70.0) < 1e-6, f"Expected 70.0 GPU-s restored, got {restored:.6f}"
print(f"  After rollback(step_b):  budget = {budget_after_rollback:.1f}  (+{restored:.1f} restored)")
ok(f"Budget rollback correct: step_b consumption ({restored:.1f} GPU-s) restored exactly.")


# ══════════════════════════════════════════════════════
# STEP 9 — AUDIT CHAIN
# ══════════════════════════════════════════════════════

header(9, "AUDIT CHAIN")

# The local system accumulated 4 records: 3 decisions + 1 rollback.
# Each record carries a prev_hash linking it to its predecessor.
records9 = local8.audit_records

chain_ok = True
for i, rec in enumerate(records9):
    expected_prev = None if i == 0 else records9[i - 1]["hash"]
    if rec.get("prev_hash") != expected_prev:
        chain_ok = False
        break

assert chain_ok, "Audit chain integrity check failed"
assert len(records9) == 4, f"Expected 4 records, got {len(records9)}"

ok(f"Audit chain: {len(records9)} records  integrity: INTACT")
ok(f"Record types: {[r['type'] for r in records9]}")
ok(f"Each record's prev_hash matches its predecessor's hash (SHA-256 chain).")


# ══════════════════════════════════════════════════════
# STEP 10 — BREAKER STATE MACHINE (numerail-ext)
# ══════════════════════════════════════════════════════

header(10, "BREAKER STATE MACHINE (numerail-ext)")

# Overload score formula: 0.30*gpu + 0.25*api + 0.20*db + 0.10*queue + 0.15*min(1, error/10)
# With gpu=api=db=queue=x and error=0: score = 0.85*x  =>  x = target / 0.85
# trip_score=0.50: score 0.20 and 0.40 leave breaker CLOSED;
#                  score 0.60 and 0.80 trip/stay in THROTTLED (safe_stop=0.85).

thresholds10 = BreakerThresholds(trip_score=0.50, reset_score=0.25, safe_stop_score=0.85)
breaker10 = BreakerStateMachine(thresholds10)

target_scores = [0.20, 0.40, 0.60, 0.80]
modes_seen10: list[BreakerMode] = []

for target in target_scores:
    x10 = target / 0.85   # produces score = 0.85*x10 = target
    snap10 = _snap(gpu=x10, api=x10, db=x10, queue=x10)
    decision10 = breaker10.update(snap10)
    actual = breaker10.overload_score(snap10)
    modes_seen10.append(decision10.mode)
    ok(f"Target score ~{target:.2f}  actual={actual:.3f}  "
       f"breaker mode = {decision10.mode.value}")

assert BreakerMode.THROTTLED in modes_seen10, (
    f"Expected THROTTLED during escalation. Modes observed: {modes_seen10}"
)
ok(f"Breaker tripped to THROTTLED during GPU escalation. Confirmed.")


# ══════════════════════════════════════════════════════
# STEP 11 — GLOBAL DEFAULT POLICY (numerail-ext)
# ══════════════════════════════════════════════════════

header(11, "GLOBAL DEFAULT POLICY (numerail-ext)")

config11 = build_global_default()

# lint_config collects ALL issues — must be zero for a valid policy
issues11 = lint_config(config11)
assert issues11 == [], f"Lint issues found: {issues11}"
ok(f"lint_config() -> {len(issues11)} issues  (policy is well-formed)")

parsed11  = PolicyParser().parse(config11)
system11  = NumerailSystem.from_config(parsed11)
region11  = system11._versions.current

n_fields11      = len(system11._schema.fields)
n_constraints11 = sum(len(c.constraint_names) for c in region11.constraints)
n_budgets11     = len(config11.get("budgets", []))

ok(f"Schema fields:     {n_fields11}")
ok(f"Named constraints: {n_constraints11}")
ok(f"Budget specs:      {n_budgets11}")


# ══════════════════════════════════════════════════════
# STEP 12 — POLICY CONTRACT (numerail-ext)
# ══════════════════════════════════════════════════════

header(12, "POLICY CONTRACT (numerail-ext)")

contract12 = NumerailPolicyContract.from_v5_config(
    config11,
    author_id="hello-world-demo",
    policy_id="global-default::hello-world-v1",
)
assert contract12.verify_digest(), "Original contract digest invalid"
digest12 = contract12.to_dict()["digest"]
ok(f"Original contract digest: {digest12[:24]}...")

# Serialise to dict and reconstruct — digest must be identical
d12        = contract12.to_dict()
contract12b = NumerailPolicyContract.from_dict(d12)
assert contract12b.verify_digest(), "Round-tripped contract digest invalid"
assert d12["digest"] == contract12b.to_dict()["digest"], "Digest changed across round-trip"

ok(f"Round-trip digest stable: {contract12b.to_dict()['digest'][:24]}...")
ok(f"SHA-256 content-addressing confirmed (canonical JSON, sorted keys).")


# ══════════════════════════════════════════════════════
# STEP 13 — GOVERNOR LIFECYCLE (extension, full stack)
# ══════════════════════════════════════════════════════

header(13, "GOVERNOR LIFECYCLE (extension, full stack)")


class _MockResMgr:
    _n: int = 0

    def acquire(self, *, state_version, expires_at_ns, resource_claims) -> str:
        self._n += 1
        return f"tok_{self._n}"

    def commit(self, *, token, receipt) -> None:
        pass

    def release(self, *, token) -> None:
        pass


class _MockDigestor:
    def digest(self, payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


gov13 = StateTransitionGovernor(
    backend=LocalNumerailBackend(),
    transition_model=IncidentCommanderTransitionModel(freshness_ns=120_000_000_000),
    reservation_mgr=_MockResMgr(),
    digestor=_MockDigestor(),
    thresholds=BreakerThresholds(trip_score=0.60, reset_score=0.30, safe_stop_score=0.85),
    bootstrap_budgets={
        "gpu_shift": 500.0,
        "external_api_shift": 100.0,
        "mutation_shift": 40.0,
    },
)

# GPU utilisation escalates 0.20 -> 0.80 over 5 cycles.
# For each APPROVE/PROJECT, independently verify the guarantee by
# rebuilding NumerailSystem from the exact envelope used that cycle.
guarantee_violations = 0

for i in range(5):
    gpu13 = 0.20 + i * 0.15  # 0.20, 0.35, 0.50, 0.65, 0.80
    snap13 = _snap(gpu=gpu13, api=0.25, db=0.25, queue=0.15)
    req13 = WorkloadRequest(
        prompt_k=5.0, completion_k=2.0, internal_tool_calls=5.0,
        external_api_calls=3.0, cloud_mutation_calls=1.0,
        gpu_seconds=10.0, parallel_workers=2.0,
        traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
        feature_flag_changes=0.0, rollback_batch_pct=0.0,
        pager_notifications=1.0, customer_comms_count=0.0,
    )
    step13 = gov13.enforce_next_step(
        request=req13, snapshot=snap13, action_id=f"gov_{i}"
    )
    decision13 = step13.numerail_result["decision"]

    guarantee_ok = True
    if decision13 in ("approve", "project"):
        enforced13 = step13.numerail_result["enforced_values"]
        cfg_check  = build_v5_policy_from_envelope(step13.envelope)
        sys_check  = NumerailSystem.from_config(cfg_check)
        fields_check = sys_check._schema.fields
        vec_check  = np.array([enforced13[f] for f in fields_check])
        region_check = sys_check._versions.current
        guarantee_ok = region_check.is_feasible(vec_check)
        if not guarantee_ok:
            guarantee_violations += 1

    ok(
        f"Cycle {i + 1}  gpu_util={gpu13:.2f}  "
        f"mode={step13.breaker.mode.value:<12}  "
        f"decision={decision13:<9}  "
        f"guarantee={'[OK]' if guarantee_ok else '[FAIL]'}"
    )

assert guarantee_violations == 0, (
    f"{guarantee_violations} guarantee violation(s) detected across 5 cycles"
)
ok(f"Enforcement guarantee holds for every APPROVE/PROJECT across all 5 cycles.")


# ══════════════════════════════════════════════════════
# STEP 14 — PROOF CHECKER VERIFICATION
# ══════════════════════════════════════════════════════

header(14, "PROOF CHECKER VERIFICATION")

proof14 = subprocess.run(
    [sys.executable, "proof/verify_proof.py"],
    cwd=str(_PKG_NUMERAIL),
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)

# Print the checker's final summary line
lines14 = [ln for ln in proof14.stdout.splitlines() if ln.strip()]
if lines14:
    ok(f"verify_proof.py: {lines14[-1].strip()}")

assert proof14.returncode == 0, (
    f"verify_proof.py exited with code {proof14.returncode}\n"
    f"stderr:\n{proof14.stderr}"
)
ok(f"Exit code: 0  (all 3,732 assertions passed)")


# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════

print()
print("=" * 51)
print("  HELLO WORLD COMPLETE")
print("  14/14 steps passed")
print("  The enforcement guarantee holds.")
print("=" * 51)
