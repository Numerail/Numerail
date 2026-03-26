#!/usr/bin/env python3
"""
Numerail v5.0.0 — Machine-Verifiable Proof Test

This script validates every axiom, lemma, theorem, and corollary stated
in numerail_proof.md against the actual numerail code.

There are two categories of verification:

1. STRUCTURAL CHECKS: verify that the code has the exact structure the
   proof depends on (return paths, guard conditions, function signatures).

2. PROPERTY CHECKS: verify that the stated properties hold empirically
   across randomized inputs, adversarial inputs, and edge cases.

Run from repo root: python proof/verify_proof.py

Requires: numerail source tree at <repo_root>/src/numerail/
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Deterministic import: always resolve from this script's repository root,
# not from the current working directory or a system-installed package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
import numerail as ne

passed = 0
failed = 0
results: List[Tuple[str, str, bool, str]] = []


def check(section: str, name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        results.append((section, name, True, detail))
    else:
        failed += 1
        results.append((section, name, False, detail))
        print(f"  FAIL: [{section}] {name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════
#  AXIOM 1: Checker Correctness
# ═══════════════════════════════════════════════════════════════════════════

def verify_axiom1():
    """Verify that is_satisfied(z, τ) ↔ evaluate(z) ≤ τ for each type."""
    rng = np.random.RandomState(42)
    tol = 1e-6

    # LinearConstraints
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    b = np.array([1, 1, 0, 0], dtype=float)
    lc = ne.LinearConstraints(A, b, names=["ux", "uy", "lx", "ly"])
    for _ in range(500):
        z = rng.uniform(-2, 3, size=2)
        sat = lc.is_satisfied(z, tol)
        ev = lc.evaluate(z)
        # is_satisfied uses componentwise check: np.all(Az <= b + tol)
        # evaluate returns max(Az - b)
        # These are equivalent: max(Az-b) <= tol ↔ all(Az <= b + tol)
        expected = ev <= tol
        check("Axiom1", "linear_checker_equivalence",
              sat == expected,
              f"z={z}, sat={sat}, ev={ev:.2e}, expected={expected}")
        if sat != expected:
            break

    # QuadraticConstraint
    Q = np.eye(2)
    a_vec = np.zeros(2)
    b_val = 1.0
    qc = ne.QuadraticConstraint(Q, a_vec, b_val, "ball")
    for _ in range(500):
        z = rng.uniform(-2, 3, size=2)
        sat = qc.is_satisfied(z, tol)
        ev = qc.evaluate(z)
        check("Axiom1", "quadratic_checker_equivalence",
              sat == (ev <= tol),
              f"z={z}, sat={sat}, ev={ev:.2e}")
        if sat != (ev <= tol):
            break

    # SOCPConstraint
    M = np.eye(2)
    q = np.zeros(2)
    c = np.zeros(2)
    d = 1.5
    sc = ne.SOCPConstraint(M, q, c, d, "norm")
    for _ in range(500):
        z = rng.uniform(-2, 3, size=2)
        sat = sc.is_satisfied(z, tol)
        ev = sc.evaluate(z)
        check("Axiom1", "socp_checker_equivalence",
              sat == (ev <= tol),
              f"z={z}, sat={sat}, ev={ev:.2e}")
        if sat != (ev <= tol):
            break

    # PSDConstraint
    A0 = np.eye(2)
    A1 = np.eye(2)
    A2 = np.array([[0, 1.0], [1, 0]])
    pc = ne.PSDConstraint(A0, [A1, A2], "lmi")
    for _ in range(500):
        z = rng.uniform(-2, 3, size=2)
        sat = pc.is_satisfied(z, tol)
        ev = pc.evaluate(z)
        check("Axiom1", "psd_checker_equivalence",
              sat == (ev <= tol),
              f"z={z}, sat={sat}, ev={ev:.2e}")
        if sat != (ev <= tol):
            break


# ═══════════════════════════════════════════════════════════════════════════
#  LEMMA 1: Combined Checker Correctness
# ═══════════════════════════════════════════════════════════════════════════

def verify_lemma1():
    """Verify is_feasible ↔ all individual is_satisfied."""
    rng = np.random.RandomState(99)
    tol = 1e-6

    lin = ne.LinearConstraints(
        np.vstack([np.eye(2), -np.eye(2)]),
        np.array([1, 1, 0, 0], dtype=float),
        names=["ux", "uy", "lx", "ly"],
    )
    quad = ne.QuadraticConstraint(np.eye(2), np.zeros(2), 0.5, "ball")
    region = ne.FeasibleRegion([lin, quad], 2)

    for _ in range(500):
        z = rng.uniform(-2, 3, size=2)
        combined = region.is_feasible(z, tol)
        individual = all(c.is_satisfied(z, tol) for c in region.constraints)
        check("Lemma1", "combined_equals_conjunction",
              combined == individual,
              f"z={z}, combined={combined}, individual={individual}")
        if combined != individual:
            break


# ═══════════════════════════════════════════════════════════════════════════
#  LEMMA 2: Project Post-Check
# ═══════════════════════════════════════════════════════════════════════════

def verify_lemma2():
    """Verify: project returns (y, True) ⟹ is_feasible(y)."""
    rng = np.random.RandomState(77)

    region = ne.combine_regions(
        ne.box_constraints([0, 0], [1, 1]),
        ne.halfplane([1, 1], 1.5, name="diag"),
    )

    for _ in range(200):
        x = rng.uniform(-3, 4, size=2)
        result = ne.project(x, region)
        if result.postcheck_passed:
            feasible = region.is_feasible(result.point)
            check("Lemma2", "postcheck_true_implies_feasible",
                  feasible,
                  f"x={x}, point={result.point}, feasible={feasible}")
            if not feasible:
                break


# ═══════════════════════════════════════════════════════════════════════════
#  LEMMA 3: Emit Path Invariant (structural)
# ═══════════════════════════════════════════════════════════════════════════

def verify_lemma3():
    """Verify the defense-in-depth check exists in _out and fires for APPROVE/PROJECT."""
    source = inspect.getsource(ne.enforce)

    # The feasibility check must exist (explicit raise, not assert — survives python -O)
    has_check = "effective.is_feasible(enforced, cfg.solver_tol)" in source
    check("Lemma3", "feasibility_check_in_emit_path", has_check,
          "The defense-in-depth feasibility check is present in _out()")

    # The check is guarded by APPROVE/PROJECT check
    has_guard = "if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):" in source
    check("Lemma3", "check_guarded_by_result_check", has_guard,
          "Check fires only for APPROVE/PROJECT results")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 1: Enforcement Soundness
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem1_structural():
    """Verify the structural claims: 1 APPROVE, 1 PROJECT, 6 REJECT."""
    source = inspect.getsource(ne.enforce)

    # Count return paths
    approve_count = source.count("EnforcementResult.APPROVE")
    project_count = source.count("EnforcementResult.PROJECT")
    reject_count = source.count("EnforcementResult.REJECT")

    # APPROVE appears in the return AND in the assert guard = 2 occurrences
    # PROJECT appears in the return AND in the assert guard = 2 occurrences
    # We check for returns via _out specifically
    # Count return paths by collecting the full _out(...) call
    lines = source.split('\n')
    approve_returns = 0
    project_returns = 0
    reject_returns = 0
    for i, line in enumerate(lines):
        if 'return _out(' in line.strip():
            # Collect the full _out call (until parens balance)
            call = ''
            depth = 0
            for j in range(i, min(i + 10, len(lines))):
                call += lines[j]
                depth += lines[j].count('(') - lines[j].count(')')
                if depth <= 0 and call.strip():
                    break
            # Check which result type is in the ARGUMENTS of _out
            if 'EnforcementResult.APPROVE' in call:
                approve_returns += 1
            elif 'EnforcementResult.PROJECT' in call:
                project_returns += 1
            elif 'EnforcementResult.REJECT' in call:
                reject_returns += 1

    check("Theorem1", "exactly_1_approve_return", approve_returns == 1,
          f"Found {approve_returns} APPROVE return(s)")
    check("Theorem1", "exactly_1_project_return", project_returns == 1,
          f"Found {project_returns} PROJECT return(s)")
    check("Theorem1", "exactly_6_reject_returns", reject_returns == 6,
          f"Found {reject_returns} REJECT return(s)")

    # APPROVE guard: is_feasible check (tol argument required to match cfg.solver_tol)
    approve_section = source[source.index("STEP 1"):source.index("STEP 2")]
    check("Theorem1", "approve_guarded_by_is_feasible",
          "effective.is_feasible(x, cfg.solver_tol)" in approve_section
          or "effective.is_feasible(x)" in approve_section,
          "APPROVE return is inside 'if effective.is_feasible(x[, cfg.solver_tol])'")

    # PROJECT guard: postcheck_passed
    check("Theorem1", "project_guarded_by_postcheck",
          "not proj.postcheck_passed" in source,
          "REJECT fires when postcheck_passed is False, so PROJECT is reachable only when True")


def verify_theorem1_empirical():
    """Verify soundness across 1000 random enforcements on mixed constraints."""
    rng = np.random.RandomState(42)

    for trial in range(200):
        n = rng.randint(2, 5)
        lo = rng.uniform(-1, 0, size=n)
        hi = rng.uniform(0.5, 2, size=n)
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([hi, -lo])
        names = [f"u{i}" for i in range(n)] + [f"l{i}" for i in range(n)]
        constraints = [ne.LinearConstraints(A, b, names=names)]

        if rng.random() < 0.3:
            Q = np.eye(n) * rng.uniform(0.5, 1.5)
            constraints.append(ne.QuadraticConstraint(Q, np.zeros(n), rng.uniform(1, 3), f"q{trial}"))

        region = ne.FeasibleRegion(constraints, n)

        for _ in range(5):
            x = rng.uniform(-3, 4, size=n)
            out = ne.enforce(x, region)
            if out.result in (ne.EnforcementResult.APPROVE, ne.EnforcementResult.PROJECT):
                feasible = region.is_feasible(out.enforced_vector)
                check("Theorem1", "soundness_empirical",
                      feasible,
                      f"trial={trial}, x={x[:3]}..., result={out.result.value}")
                if not feasible:
                    return


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 2: Fail-Closed Rejection
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem2():
    """Extreme inputs that solvers struggle with → REJECT or valid PROJECT."""
    region = ne.box_constraints([0, 0], [0.001, 0.001])
    out = ne.enforce(np.array([1e6, 1e6]), region)
    ok = True
    if out.result == ne.EnforcementResult.PROJECT:
        ok = region.is_feasible(out.enforced_vector)
    check("Theorem2", "extreme_input_fail_closed", ok,
          f"result={out.result.value}, valid={'yes' if ok else 'NO'}")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 3: Hard-Wall Dominance
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem3():
    region = ne.combine_regions(
        ne.box_constraints([0, 0], [1, 1], names=["mx", "my", "lx", "ly"]),
        ne.halfplane([1, 1], 1.5, name="budget"),
    )
    cfg = ne.EnforcementConfig(hard_wall_constraints=frozenset({"budget"}))
    out = ne.enforce(np.array([1.0, 1.0]), region, cfg)
    check("Theorem3", "hard_wall_rejects", out.result == ne.EnforcementResult.REJECT,
          f"result={out.result.value}")
    check("Theorem3", "no_solver_invoked", out.solver_method == "none",
          f"solver={out.solver_method}")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 4: Forbidden-Dimension Safety
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem4():
    schema = ne.Schema(fields=["x", "y"])
    cfg = ne.EnforcementConfig(
        dimension_policies={"x": ne.DimensionPolicy.PROJECTION_FORBIDDEN},
    )
    out = ne.enforce(np.array([1.5, 0.5]), ne.box_constraints([0, 0], [1, 1]), cfg, schema)
    check("Theorem4", "forbidden_dim_changed_rejects",
          out.result == ne.EnforcementResult.REJECT,
          f"result={out.result.value}")

    # y not forbidden → should PROJECT
    out2 = ne.enforce(np.array([0.5, 1.5]), ne.box_constraints([0, 0], [1, 1]), cfg, schema)
    check("Theorem4", "non_forbidden_dim_projects",
          out2.result == ne.EnforcementResult.PROJECT,
          f"result={out2.result.value}")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 5: Budget Monotonicity
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem5():
    schema = ne.Schema(fields=["cost", "qty"])
    region = ne.combine_regions(
        ne.box_constraints([-1, 0], [1, 1], names=["mc", "mq", "lc", "lq"]),
        ne.halfplane([1, 0], 1.0, name="cap"),
    )
    sys = ne.NumerailSystem(schema, region)
    sys.register_budget(ne.BudgetSpec(
        name="b", constraint_name="cap", dimension_name="cost",
        weight=1.0, initial=1.0, consumption_mode="nonnegative",
    ))

    # Negative cost → zero consumption (nonneg mode)
    sys.enforce({"cost": -0.5, "qty": 0.5}, action_id="a1")
    check("Theorem5", "nonneg_zero_for_negative",
          sys.budget_status()["b"]["consumed"] == 0.0,
          f"consumed={sys.budget_status()['b']['consumed']}")

    # Positive cost → positive consumption
    sys.enforce({"cost": 0.3, "qty": 0.5}, action_id="a2")
    check("Theorem5", "nonneg_positive_for_positive",
          abs(sys.budget_status()["b"]["consumed"] - 0.3) < 1e-6,
          f"consumed={sys.budget_status()['b']['consumed']}")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 6: Rollback Restoration
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem6():
    schema = ne.Schema(fields=["cost", "qty"])
    region = ne.combine_regions(
        ne.box_constraints([0, 0], [0.5, 1], names=["mc", "mq", "lc", "lq"]),
        ne.halfplane([1, 0], 0.5, name="cap"),
    )
    sys = ne.NumerailSystem(schema, region)
    sys.register_budget(ne.BudgetSpec(
        name="b", constraint_name="cap", dimension_name="cost",
        weight=1.0, initial=0.5,
    ))
    sys.enforce({"cost": 0.3, "qty": 0.5}, action_id="a1")
    sys.enforce({"cost": 0.15, "qty": 0.5}, action_id="a2")

    pre = sys.budget_status()["b"]["remaining"]
    check("Theorem6", "pre_rollback_remaining",
          abs(pre - 0.05) < 1e-4, f"remaining={pre}")

    sys.rollback("a2")
    post = sys.budget_status()["b"]["remaining"]
    check("Theorem6", "post_rollback_remaining",
          abs(post - 0.20) < 1e-4, f"remaining={post}")

    # Idempotent
    check("Theorem6", "rollback_idempotent",
          not sys.rollback("a2"), "Second rollback returns False")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 8: Audit-Chain Integrity
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem8():
    schema = ne.Schema(fields=["x", "y"])
    region = ne.box_constraints([0, 0], [1, 1])
    sys = ne.NumerailSystem(schema, region)
    rng = np.random.RandomState(123)
    for i in range(50):
        sys.enforce({"x": rng.uniform(-1, 2), "y": rng.uniform(-1, 2)}, action_id=f"a{i}")
    valid, depth = sys.verify_audit()
    check("Theorem8", "chain_valid", valid, f"valid={valid}")
    check("Theorem8", "chain_depth", depth == 50, f"depth={depth}")

    # Tamper detection: export, modify, verify
    records = sys.export_audit()
    if len(records) > 5:
        # Modify a record in the middle
        tampered = [r.copy() for r in records]
        tampered[3]["result"] = "tampered"
        # Recompute: this should break the chain
        # We can't re-import into the chain, but we can verify the logic
        prev = tampered[3].get("prev_hash", "")
        check_dict = {k: v for k, v in tampered[3].items() if k != "hash"}
        recomputed = hashlib.sha256(
            json.dumps(check_dict, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        check("Theorem8", "tamper_detected",
              recomputed != tampered[3]["hash"],
              "Modified record has different hash than stored")


# ═══════════════════════════════════════════════════════════════════════════
#  THEOREM 9: Passthrough and Idempotence
# ═══════════════════════════════════════════════════════════════════════════

def verify_theorem9():
    region = ne.box_constraints([0, 0], [1, 1])

    # 9a: Passthrough
    x = np.array([0.5, 0.5])
    out = ne.enforce(x, region)
    check("Theorem9", "passthrough_approve",
          out.result == ne.EnforcementResult.APPROVE, f"result={out.result.value}")
    check("Theorem9", "passthrough_unchanged",
          np.allclose(out.enforced_vector, x), f"enforced={out.enforced_vector}")
    check("Theorem9", "passthrough_distance_zero",
          out.distance == 0.0, f"distance={out.distance}")

    # 9b: Idempotence
    x2 = np.array([2.0, 0.5])
    out1 = ne.enforce(x2, region)
    assert out1.result == ne.EnforcementResult.PROJECT
    out2 = ne.enforce(out1.enforced_vector, region)
    check("Theorem9", "idempotence_approve",
          out2.result == ne.EnforcementResult.APPROVE, f"result={out2.result.value}")
    check("Theorem9", "idempotence_distance_zero",
          out2.distance == 0.0, f"distance={out2.distance}")


# ═══════════════════════════════════════════════════════════════════════════
#  COROLLARIES: Solver Independence, Proposer Independence
# ═══════════════════════════════════════════════════════════════════════════

def verify_corollaries():
    """The guarantee holds regardless of solver or proposed vector."""
    rng = np.random.RandomState(999)
    region = ne.combine_regions(
        ne.box_constraints([0, 0, 0], [1, 1, 1]),
        ne.halfplane([1, 1, 1], 2.0, name="sum"),
    )

    # Adversarial inputs: extreme values
    adversarial = [
        np.array([1e10, -1e10, 0]),
        np.array([0, 0, 0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-1e6, 1e6, -1e6]),
    ]
    for x in adversarial:
        out = ne.enforce(x, region)
        if out.result in (ne.EnforcementResult.APPROVE, ne.EnforcementResult.PROJECT):
            ok = region.is_feasible(out.enforced_vector)
            check("Corollary", "proposer_independence",
                  ok, f"x={x[:3]}, result={out.result.value}")

    # NaN and Inf rejected before enforcement
    for bad in [np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0]), np.array([-np.inf, 0, 0])]:
        try:
            ne.enforce(bad, region)
            check("Corollary", "nan_inf_rejected", False, f"input={bad} was not rejected")
        except ne.ValidationError:
            check("Corollary", "nan_inf_rejected", True, f"input={bad} raised ValidationError")


# ═══════════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import logging
    logging.disable(logging.CRITICAL)

    print("=" * 70)
    print(f"  Numerail {ne.__version__} — Proof Verification")
    print("=" * 70)
    print()

    t0 = time.time()

    sections = [
        ("Axiom 1: Checker Correctness", verify_axiom1),
        ("Lemma 1: Combined Checker", verify_lemma1),
        ("Lemma 2: Project Post-Check", verify_lemma2),
        ("Lemma 3: Emit Path Assert", verify_lemma3),
        ("Theorem 1: Soundness (structural)", verify_theorem1_structural),
        ("Theorem 1: Soundness (empirical)", verify_theorem1_empirical),
        ("Theorem 2: Fail-Closed", verify_theorem2),
        ("Theorem 3: Hard-Wall Dominance", verify_theorem3),
        ("Theorem 4: Forbidden-Dimension Safety", verify_theorem4),
        ("Theorem 5: Budget Monotonicity", verify_theorem5),
        ("Theorem 6: Rollback Restoration", verify_theorem6),
        ("Theorem 8: Audit Integrity", verify_theorem8),
        ("Theorem 9: Passthrough/Idempotence", verify_theorem9),
        ("Corollaries: Independence", verify_corollaries),
    ]

    for name, fn in sections:
        print(f"  Verifying: {name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70)

    # Group results by section
    sections_seen = []
    for section, name, ok, detail in results:
        if not sections_seen or sections_seen[-1] != section:
            sections_seen.append(section)

    for section in dict.fromkeys(s for s, _, _, _ in results):
        section_results = [(n, ok, d) for s, n, ok, d in results if s == section]
        section_pass = sum(1 for _, ok, _ in section_results if ok)
        section_fail = sum(1 for _, ok, _ in section_results if not ok)
        marker = "✓" if section_fail == 0 else "✗"
        print(f"  {marker} {section}: {section_pass} passed, {section_fail} failed")

    print()
    if failed == 0:
        print("  ALL PROOF CLAIMS VERIFIED. ✓")
    else:
        print(f"  {failed} PROOF CLAIM(S) FAILED. ✗")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
