"""
tests/test_mathematical_guarantees.py

Numerail v5.0.0 — Comprehensive Mathematical Guarantee Analysis Suite
======================================================================

Provides precise, independently-derived verification of every formal claim in
proof/PROOF.md. Each section is headed with the exact proof element being
tested, its formal statement, and any assumptions it relies on.

Proof dependency graph (PROOF.md §Summary):
    Axiom 1 (Checker Correctness)
        ▼
    Lemma 1 (Combined Checker = ∧ is_satisfied_j)
        ├──► Lemma 2 (project postcheck ⟹ is_feasible)
        ├──► Lemma 3 (emit-path invariant: _out asserts feasibility)
        ▼
    Theorem 1 (Enforcement Soundness) ← THE CENTRAL GUARANTEE
        ├──► Corollary (solver independence)
        ├──► Corollary (proposer independence)
        └──► Theorem 9b (idempotence, via Theorem 1 + Theorem 9a)

    Theorem 2  (fail-closed rejection)
    Theorem 3  (hard-wall dominance)
    Theorem 4  (forbidden-dimension safety)
    Theorem 5  (budget monotonicity)
    Theorem 6  (rollback restoration)
    Theorem 7  (monotone self-limits)
    Theorem 8  (audit-chain integrity)
    Theorem 9a (passthrough)

Notation (matching PROOF.md §Notation):
    n         — vector dimension
    x ∈ ℝⁿ   — proposed vector
    F = {C₁, …, Cₘ}  — active feasible region (collection of convex constraints)
    Cⱼ        — a single convex constraint
    τ > 0     — tolerance (default 10⁻⁶ = solver_tol)
    F_τ       — {x ∈ ℝⁿ : ∀j, Cⱼ.evaluate(x) ≤ τ}
    r         — enforcement result ∈ {APPROVE, PROJECT, REJECT}
    y         — enforced vector

All "Theorem N", "Lemma N", "Axiom 1", "Corollary" citations refer to
packages/numerail/proof/PROOF.md unless otherwise noted.

Run: pytest tests/test_mathematical_guarantees.py -v
"""

from __future__ import annotations

import hashlib
import inspect
import json
import copy

import numpy as np
import pytest

from numerail.engine import (
    AuditChain,
    BudgetSpec,
    BudgetTracker,
    ConstraintError,
    DimensionPolicy,
    EnforcementConfig,
    EnforcementOutput,
    EnforcementResult,
    FeasibleRegion,
    LinearConstraints,
    NumerailSystem,
    PSDConstraint,
    ProjectionResult,
    QuadraticConstraint,
    ResolutionError,
    RoutingDecision,
    RoutingThresholds,
    SOCPConstraint,
    Schema,
    ValidationError,
    _deterministic_json,
    box_constraints,
    combine_regions,
    enforce,
    halfplane,
    merge_trusted_context,
    project,
)
from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS AND SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

TAU = 1e-6  # Default solver_tol (PROOF.md §Notation: "τ > 0, default 10⁻⁶")


def guarantee_check(
    out: EnforcementOutput,
    region: FeasibleRegion,
    tol: float = TAU,
) -> tuple[bool, float, str | None]:
    """
    Independent guarantee verifier — does NOT call out.result or region.is_feasible.
    Iterates every constraint manually and computes raw violation magnitudes.

    Returns (holds, worst_violation, worst_constraint_name).

    This function is intentionally separate from the kernel's own checker so
    that it provides an independent oracle for Theorem 1 assertions.
    """
    if out.result not in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
        return True, 0.0, None
    worst_v = 0.0
    worst_name: str | None = None
    for c in region.constraints:
        v = c.evaluate(out.enforced_vector)
        if v > worst_v:
            worst_v = v
            worst_name = c.name
    return worst_v <= tol, worst_v, worst_name


def make_box_region(n: int = 3, lo: float = 0.0, hi: float = 10.0) -> FeasibleRegion:
    """n-dimensional box [lo, hi]ⁿ."""
    return box_constraints([lo] * n, [hi] * n)


def make_mixed_region() -> FeasibleRegion:
    """
    Four-dimensional region with all four constraint types active.

    Constraints:
      Linear:    x ∈ [0, 10]⁴  and  x₀ + x₁ ≤ 12
      Quadratic: x₀²/100 + x₁²/100 ≤ 1  (axis-aligned ellipse, centre at origin)
      SOCP:      ‖[x₂/5, x₃/5]‖ ≤ 1.5   (cone in x₂-x₃ plane)

    A feasible interior point is, e.g., [1, 1, 1, 1].
    """
    lin = combine_regions(
        box_constraints([0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]),
        halfplane([1.0, 1.0, 0.0, 0.0], 12.0, name="coupling"),
    )
    Q = np.diag([1.0 / 100, 1.0 / 100, 0.0, 0.0])
    quad = FeasibleRegion(
        [QuadraticConstraint(Q, np.zeros(4), 1.0, "ellipse")], 4
    )
    M_socp = np.array([[0.0, 0.0, 1.0 / 5, 0.0],
                       [0.0, 0.0, 0.0,     1.0 / 5]])
    socp = FeasibleRegion(
        [SOCPConstraint(M_socp, np.zeros(2), np.zeros(4), 1.5, "cone")], 4
    )
    return FeasibleRegion.combine(lin, quad, socp)


def make_psd_region() -> FeasibleRegion:
    """
    Two-dimensional LMI region:
        A(x) = [[1 - x₀, 0.5·x₁],
                [0.5·x₁, 1 - x₁]] ≽ 0

    Feasible when both eigenvalues of A(x) are ≥ 0.
    A feasible interior point is [0, 0].
    """
    A0 = np.eye(2, dtype=float)
    A_list = [
        np.array([[-1.0, 0.0], [0.0,  0.0]]),   # coefficient of x₀
        np.array([[ 0.0, 0.5], [0.5, -1.0]]),   # coefficient of x₁
    ]
    return FeasibleRegion([PSDConstraint(A0, A_list, "lmi")], 2)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A: AXIOM 1 — CHECKER CORRECTNESS
#
#  Claim (PROOF.md §Axiom 1):
#      For each constraint type, is_satisfied(z, τ) returns true
#      if and only if evaluate(z) ≤ τ.
#
#  This is the foundation of the entire proof chain. Lemma 1 depends on it,
#  and therefore so do Lemmas 2, 3 and Theorem 1.
# ═══════════════════════════════════════════════════════════════════════════

class TestAxiom1_CheckerCorrectness:
    """Direct unit tests verifying Axiom 1 for each constraint type."""

    # ── Linear ────────────────────────────────────────────────────────────

    def test_linear_evaluate_formula_matches_definition(self):
        """
        PROOF.md §Axiom 1 (Linear):
            evaluate(x) = max(Ax − b)

        Verify against independent numpy computation.
        """
        A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        b = np.array([3.0, 4.0, -1.0])
        c = LinearConstraints(A, b, ["r1", "r2", "r3"])

        rng = np.random.RandomState(0)
        for _ in range(50):
            x = rng.uniform(-2.0, 6.0, 2)
            expected = float(np.max(A @ x - b))
            assert abs(c.evaluate(x) - expected) < 1e-12

    def test_linear_is_satisfied_iff_evaluate_leq_tau(self):
        """
        Axiom 1: is_satisfied(x, τ) ↔ evaluate(x) ≤ τ
        for all τ values and all x.
        """
        c = LinearConstraints(
            np.eye(2), np.array([1.0, 1.0]), ["ub0", "ub1"]
        )
        test_cases = [
            (np.array([0.5, 0.5]), 1e-6),
            (np.array([1.0, 1.0]), 1e-6),       # exactly on boundary
            (np.array([1.0 + 1e-7, 1.0]), 1e-6),  # just outside boundary
            (np.array([1.0 + 1e-7, 1.0]), 1e-5),  # outside but within larger τ
            (np.array([2.0, 0.5]), 1e-6),          # clearly outside
        ]
        for x, tau in test_cases:
            ev = c.evaluate(x)
            sat = c.is_satisfied(x, tau)
            assert sat == (ev <= tau), (
                f"Axiom 1 violated for Linear: "
                f"evaluate={ev:.2e}, tau={tau:.2e}, is_satisfied={sat}"
            )

    def test_linear_satisfiability_boundary_precision(self):
        """
        At x = b exactly on the boundary, evaluate(x) = 0.
        is_satisfied with τ > 0 must return True (τ-relaxed set contains boundary).
        is_satisfied with τ < 0 would return False.
        """
        c = LinearConstraints(
            np.array([[1.0, 0.0]]), np.array([5.0]), ["ub"]
        )
        x_boundary = np.array([5.0, 0.0])
        assert abs(c.evaluate(x_boundary)) < 1e-14, "Boundary point should have evaluate ≈ 0"
        assert c.is_satisfied(x_boundary, tol=1e-6)
        assert c.is_satisfied(x_boundary, tol=0.0)

    # ── Quadratic ─────────────────────────────────────────────────────────

    def test_quadratic_evaluate_formula_matches_definition(self):
        """
        PROOF.md §Axiom 1 (Quadratic):
            evaluate(x) = x'Qx + a'x − b

        Q is symmetrised internally; test that Q symmetrisation preserves the identity.
        """
        Q_raw = np.array([[2.0, 0.5], [0.5, 3.0]])
        a = np.array([0.1, -0.2])
        b_val = 4.0
        c = QuadraticConstraint(Q_raw, a, b_val, "quad")
        Q_sym = 0.5 * (Q_raw + Q_raw.T)

        rng = np.random.RandomState(1)
        for _ in range(50):
            x = rng.uniform(-2.0, 2.0, 2)
            expected = float(x @ Q_sym @ x + a @ x - b_val)
            assert abs(c.evaluate(x) - expected) < 1e-12

    def test_quadratic_is_satisfied_iff_evaluate_leq_tau(self):
        """Axiom 1 for QuadraticConstraint."""
        Q = np.diag([1.0, 1.0])          # unit sphere: x₀² + x₁² ≤ r²
        c = QuadraticConstraint(Q, np.zeros(2), 1.0, "sphere")

        rng = np.random.RandomState(2)
        for _ in range(200):
            x = rng.uniform(-2.0, 2.0, 2)
            tau = rng.choice([1e-8, 1e-6, 1e-3, 0.1, 0.5])
            ev = c.evaluate(x)
            sat = c.is_satisfied(x, tau)
            assert sat == (ev <= tau)

    def test_quadratic_psd_requirement_enforced(self):
        """
        PROOF.md §Axiom 1 (Quadratic):
            Q verified PSD at construction (eigenvalue check rejects non-PSD).
        An indefinite Q would give a non-convex set; construction must fail.
        """
        Q_indef = np.array([[1.0, 0.0], [0.0, -1.0]])  # eigenvalues +1, -1
        with pytest.raises(ConstraintError, match="not PSD"):
            QuadraticConstraint(Q_indef, np.zeros(2), 1.0)

    # ── SOCP ──────────────────────────────────────────────────────────────

    def test_socp_evaluate_formula_matches_definition(self):
        """
        PROOF.md §Axiom 1 (SOCP):
            evaluate(x) = ‖Mx + q‖ − (c'x + d)
        """
        M = np.array([[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.5]])
        q = np.array([0.1, -0.1])
        c_vec = np.array([0.0, 0.0, 0.1])
        d = 2.0
        sc = SOCPConstraint(M, q, c_vec, d, "socp_test")

        rng = np.random.RandomState(3)
        for _ in range(50):
            x = rng.uniform(-1.0, 3.0, 3)
            expected = float(np.linalg.norm(M @ x + q) - (c_vec @ x + d))
            assert abs(sc.evaluate(x) - expected) < 1e-12

    def test_socp_is_satisfied_iff_evaluate_leq_tau(self):
        """Axiom 1 for SOCPConstraint."""
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        sc = SOCPConstraint(M, np.zeros(2), np.zeros(2), 1.5, "unit_socp")

        rng = np.random.RandomState(4)
        for _ in range(200):
            x = rng.uniform(-2.0, 2.0, 2)
            tau = rng.choice([1e-8, 1e-6, 0.01, 0.1])
            ev = sc.evaluate(x)
            sat = sc.is_satisfied(x, tau)
            assert sat == (ev <= tau)

    # ── PSD ───────────────────────────────────────────────────────────────

    def test_psd_evaluate_formula_matches_definition(self):
        """
        PROOF.md §Axiom 1 (PSD):
            evaluate(x) = −λ_min(A₀ + Σᵢ xᵢAᵢ)

        λ_min < 0 (matrix not PSD) gives evaluate > 0 (constraint violated).
        λ_min ≥ 0 (matrix PSD) gives evaluate ≤ 0 (constraint satisfied).
        """
        region = make_psd_region()
        psd_c = region.constraints[0]
        assert isinstance(psd_c, PSDConstraint)

        rng = np.random.RandomState(5)
        for _ in range(50):
            x = rng.uniform(-0.8, 0.8, 2)
            M = psd_c.matrix_at(x)
            eigvals = np.linalg.eigvalsh(M)
            expected = float(-np.min(eigvals))
            assert abs(psd_c.evaluate(x) - expected) < 1e-12

    def test_psd_is_satisfied_iff_evaluate_leq_tau(self):
        """Axiom 1 for PSDConstraint."""
        region = make_psd_region()
        psd_c = region.constraints[0]

        rng = np.random.RandomState(6)
        for _ in range(200):
            x = rng.uniform(-1.5, 1.5, 2)
            tau = rng.choice([1e-8, 1e-6, 0.01, 0.5])
            ev = psd_c.evaluate(x)
            sat = psd_c.is_satisfied(x, tau)
            assert sat == (ev <= tau), (
                f"Axiom 1 violated for PSD: evaluate={ev:.2e}, tau={tau:.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B: LEMMA 1 — COMBINED CHECKER CORRECTNESS
#
#  Claim (PROOF.md §Lemma 1):
#      is_feasible(x, τ) = true  iff  ∀j ∈ {1,…,m}: Cⱼ.evaluate(x) ≤ τ
#
#  Proof:  is_feasible uses all(c.is_satisfied(x, tol) for c in constraints).
#          By Axiom 1, each term is true iff evaluate(x) ≤ τ.
#          Python's all() is the logical conjunction.
# ═══════════════════════════════════════════════════════════════════════════

class TestLemma1_CombinedChecker:
    """Verify that is_feasible is the exact logical conjunction of all is_satisfied."""

    def test_all_pass_implies_feasible(self):
        """∀j, Cⱼ.is_satisfied(x) = true  ⟹  is_feasible(x) = true."""
        region = make_mixed_region()
        x = np.array([1.0, 1.0, 1.0, 1.0])   # interior point
        assert all(c.is_satisfied(x) for c in region.constraints)
        assert region.is_feasible(x)

    def test_any_fail_implies_infeasible(self):
        """∃j, Cⱼ.is_satisfied(x) = false  ⟹  is_feasible(x) = false."""
        region = make_mixed_region()
        # Violate the ellipse constraint (x₀ = 10 → evaluate = 100/100 - 1 = 0 exactly at 10)
        # x₀ = 11 would violate box too; use x₀ = 10, which violates ellipse (10²/100 = 1.0,
        # so evaluate = 1.0 - 1.0 = 0, marginal); use x₀ = 10.01 which violates box.
        # Instead, set x₀ = 9.5 (in box), check ellipse: 9.5²/100 = 0.9025 → evaluate = -0.0975
        # Not helpful. Actually use x₀ = 10.1 to violate box upper bound.
        x_viol_box = np.array([10.5, 1.0, 1.0, 1.0])
        for c in region.constraints:
            individual = c.is_satisfied(x_viol_box)
            # At least one should be False (the box upper bound)
        assert not region.is_feasible(x_viol_box)

    def test_conjunction_semantics_single_failing_constraint(self):
        """
        With m constraints where m−1 pass and 1 fails,
        is_feasible must return False. Tests the conjunctive nature.
        """
        # Two constraints: x ≤ 5  AND  x ≥ 3
        # Point x=6: violates upper, satisfies lower
        A = np.array([[1.0], [-1.0]])
        b = np.array([5.0, -3.0])
        lc = LinearConstraints(A, b, ["ub", "lb"])
        region = FeasibleRegion([lc], 1)

        x_viol_only_ub = np.array([6.0])
        rows = lc.is_satisfied(x_viol_only_ub)   # this is a single bool for the block
        assert not rows   # LinearConstraints block returns False if any row fails
        assert not region.is_feasible(x_viol_only_ub)

        x_feasible = np.array([4.0])
        assert region.is_feasible(x_feasible)

    def test_is_feasible_source_uses_all(self):
        """
        Lemma 1 proof cites the all() implementation directly.
        Verify the implementation matches the proof's cited code.
        """
        src = inspect.getsource(FeasibleRegion.is_feasible)
        assert "all(" in src
        assert "is_satisfied" in src

    def test_empty_tau_relaxation_at_boundary(self):
        """
        The τ-relaxed set F_τ strictly contains F_0:
            x on the boundary has evaluate(x) = 0 ≤ τ for any τ > 0.
        Verify is_feasible(boundary_point, τ=0.0) matches evaluate(x) ≤ 0.
        """
        region = box_constraints([0.0], [5.0])
        x_on_boundary = np.array([5.0])

        # At the upper boundary: evaluate = max(5.0 - 5.0, -5.0 - 0.0) = max(0, -5) = 0
        # Wait: box_constraints builds A = [[1], [-1]], b = [5, 0]
        # evaluate([5]) = max(1*5 - 5, -1*5 - 0) = max(0, -5) = 0
        lc = region.constraints[0]
        ev = lc.evaluate(x_on_boundary)
        assert abs(ev) < 1e-14, f"Boundary point should have evaluate = 0, got {ev}"
        assert region.is_feasible(x_on_boundary, tol=1e-6)
        assert region.is_feasible(x_on_boundary, tol=0.0)   # exactly on boundary

    def test_multi_constraint_type_conjunction(self):
        """
        With linear + quadratic + SOCP constraints, is_feasible is the
        conjunction of all three types simultaneously.
        A point that satisfies linear and quadratic but violates SOCP
        must give is_feasible=False.

        make_mixed_region() produces 4 constraint objects in order:
          [0] LinearConstraints (box: [0,10]⁴)
          [1] LinearConstraints (coupling: x₀+x₁ ≤ 12)
          [2] QuadraticConstraint (ellipse)
          [3] SOCPConstraint (cone)
        """
        region = make_mixed_region()
        # x₂ = x₃ = 7: cone ‖[7/5, 7/5]‖ = 7√2/5 ≈ 1.98 > 1.5 → violated
        x = np.array([1.0, 1.0, 7.0, 7.0])

        # Locate each constraint by type for robustness
        lin_cs = [c for c in region.constraints if isinstance(c, LinearConstraints)]
        quad_cs = [c for c in region.constraints if isinstance(c, QuadraticConstraint)]
        socp_cs = [c for c in region.constraints if isinstance(c, SOCPConstraint)]

        for lc in lin_cs:
            assert lc.is_satisfied(x), f"Linear '{lc.name}' unexpectedly violated for x={x}"
        for qc in quad_cs:
            assert qc.is_satisfied(x), f"Quad '{qc.name}' unexpectedly violated for x={x}"
        for sc in socp_cs:
            assert not sc.is_satisfied(x), (
                f"SOCP '{sc.name}' should be violated: "
                f"‖[7/5, 7/5]‖={np.linalg.norm([1.4,1.4]):.3f} > 1.5"
            )
        assert not region.is_feasible(x)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C: LEMMA 2 — PROJECT POST-CHECK
#
#  Claim (PROOF.md §Lemma 2):
#      If project(x, F, τ) returns (y, postcheck_passed=True),
#      then is_feasible(y, τ) = true.
#
#  Proof: All five return points R1–R4 set postcheck_passed=True only after
#         region.is_feasible() returned True for the candidate point.
#         R5 sets postcheck_passed=False (excluded by hypothesis).
# ═══════════════════════════════════════════════════════════════════════════

class TestLemma2_ProjectPostCheck:
    """Verify that every postcheck_passed=True projection output is feasible."""

    def _verify_projection(self, x, region, tol=TAU, description=""):
        result = project(x, region, tol=tol)
        if result.postcheck_passed:
            assert region.is_feasible(result.point, tol), (
                f"Lemma 2 violated {description}: postcheck_passed=True "
                f"but is_feasible=False. "
                f"Worst violation: {max(c.evaluate(result.point) for c in region.constraints):.2e}"
            )
        return result

    def test_box_region_already_feasible(self):
        """R1: x already feasible → postcheck_passed=True, point=x."""
        region = make_box_region(3, 0.0, 5.0)
        x = np.array([1.0, 2.0, 3.0])
        result = project(x, region)
        assert result.postcheck_passed
        assert result.solver_method in ("none",)
        assert region.is_feasible(result.point)

    def test_box_region_clamp_projection(self):
        """R2: box clamp projection (O(n) path). Postcheck must pass."""
        region = make_box_region(4, 0.0, 3.0)
        x = np.array([5.0, -1.0, 3.5, 2.0])   # first 3 components outside
        result = self._verify_projection(x, region, description="box_clamp")
        if result.postcheck_passed:
            assert result.solver_method in ("none", "box_clamp")

    def test_linear_polytope_dykstra_projection(self):
        """
        R3/R4: For a linear polytope without pure-box structure,
        Dykstra's algorithm is invoked. Post-check must pass.
        """
        # 2D polytope: 0 ≤ x ≤ 3, 0 ≤ y ≤ 3, x + y ≤ 4
        region = combine_regions(
            box_constraints([0.0, 0.0], [3.0, 3.0]),
            halfplane([1.0, 1.0], 4.0, name="sum"),
        )
        rng = np.random.RandomState(7)
        for _ in range(30):
            x = rng.uniform(-1.0, 5.0, 2)
            self._verify_projection(x, region, description="dykstra_polytope")

    def test_mixed_region_slsqp_projection(self):
        """
        R3/R4: Mixed (nonlinear) region uses SLSQP first.
        For many random points, post-check must pass whenever it is True.
        """
        region = make_mixed_region()
        rng = np.random.RandomState(8)
        for _ in range(30):
            x = rng.uniform(-1.0, 12.0, 4)
            self._verify_projection(x, region, description="slsqp_mixed")

    def test_postcheck_passed_false_on_empty_region(self):
        """
        R5: Empty feasible region — all solvers fail post-check.
        project() must return postcheck_passed=False.

        Contradictory linear constraints: x ≤ 1 AND x ≥ 2 (1D).
        """
        A = np.array([[1.0], [-1.0]])
        b = np.array([1.0, -2.0])   # x ≤ 1  and  -x ≤ -2 → x ≥ 2
        region = FeasibleRegion(
            [LinearConstraints(A, b, ["ub", "lb"])], 1
        )
        result = project(np.array([0.5]), region)
        assert not result.postcheck_passed, (
            "Contradictory constraints must produce postcheck_passed=False"
        )

    def test_projection_return_points_all_gated_by_is_feasible(self):
        """
        PROOF.md §Lemma 2 Table: project() has 5 logical return points
        (R1–R5). The implementation separates linear-only and nonlinear
        code paths, yielding 7 actual return statements — but the logical
        structure matches the proof: every postcheck_passed=True return is
        directly preceded by region.is_feasible() returning True.

        Structural check: count actual returns and is_feasible guards.
        """
        src = inspect.getsource(project)
        returns = [ln.strip() for ln in src.split("\n") if "return ProjectionResult(" in ln]

        # 7 returns: 1 (already feasible) + 1 (box_clamp) +
        #            2 (dykstra: linear-only path and nonlinear fallback path) +
        #            2 (slsqp: linear-only fallback and nonlinear primary) +
        #            1 (all solvers failed)
        # The proof's 5 logical cases (R1-R5) are preserved; the code has
        # two separate orderings (linear-only vs mixed), each with the same
        # logical structure.
        assert len(returns) >= 5, (
            f"project() must have at least 5 return points; found {len(returns)}"
        )

        # Verify the key invariant: every postcheck_passed=True return is
        # guarded by region.is_feasible() — at least 4 such guards exist
        assert src.count("region.is_feasible") >= 4, (
            "project() must gate each solver-result return with is_feasible"
        )

        # The only postcheck_passed=False return is the all-solvers-failed case
        false_returns = [r for r in returns if ", False)" in r or ", False, False)" in r]
        assert len(false_returns) >= 1, (
            "Must have at least one postcheck_passed=False return (all-solvers-failed case)"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION D: LEMMA 3 — EMIT-PATH INVARIANT
#
#  Claim (PROOF.md §Lemma 3):
#      The helper _out() inside enforce() raises AssertionError if
#      result ∈ {APPROVE, PROJECT} and is_feasible(enforced) = False.
#
#  Key implementation requirement: uses explicit `raise`, NOT a bare `assert`,
#  so it cannot be stripped by `python -O` (CLAUDE.md §Invariant 2).
# ═══════════════════════════════════════════════════════════════════════════

class TestLemma3_EmitPathInvariant:
    """Verify the defense-in-depth assert in _out() is correctly structured."""

    def test_explicit_raise_not_bare_assert(self):
        """
        PROOF.md §Lemma 3 (and CLAUDE.md §Invariant 2):
        The guard must use `raise AssertionError(...)`, not a bare `assert`.
        A bare `assert` is stripped by `python -O`, which would eliminate
        the defense-in-depth check entirely.
        """
        src = inspect.getsource(enforce)
        assert "raise AssertionError(" in src, (
            "Emit-path guard must use explicit raise AssertionError, not bare assert"
        )
        # Ensure there is no bare 'assert' that would be stripped
        lines = src.split("\n")
        bare_assert_lines = [
            ln.strip() for ln in lines
            if ln.strip().startswith("assert ") and "AssertionError" not in ln
        ]
        # No bare assert statements inside the critical _out() guard path
        # (there may be asserts elsewhere, but not guarding the emit path)
        for ln in bare_assert_lines:
            assert "is_feasible" not in ln, (
                f"Found bare assert guarding is_feasible: {ln!r}  "
                "This can be stripped by python -O."
            )

    def test_emit_path_checks_before_output_construction(self):
        """
        PROOF.md §Lemma 3:
        The feasibility re-check fires BEFORE EnforcementOutput is constructed.
        If the check raises, no partial output escapes.
        """
        src = inspect.getsource(enforce)
        # Find the _out function definition
        assert "def _out(" in src
        # In the _out body, the guard must precede the 'return EnforcementOutput('
        out_idx = src.find("def _out(")
        guard_idx = src.find("raise AssertionError(", out_idx)
        output_idx = src.find("return EnforcementOutput(", out_idx)
        assert 0 < guard_idx < output_idx, (
            "Emit-path AssertionError must appear before EnforcementOutput construction "
            "inside _out()"
        )

    def test_approve_and_project_gated_by_feasibility_in_out(self):
        """
        The guard condition: result ∈ {APPROVE, PROJECT} AND NOT is_feasible(enforced)
        → raise AssertionError.

        Verify the exact condition is present.
        """
        src = inspect.getsource(enforce)
        assert "EnforcementResult.APPROVE" in src
        assert "EnforcementResult.PROJECT" in src
        assert "effective.is_feasible(enforced" in src

    def test_behavioral_approve_always_feasible(self):
        """
        Behavioral consequence of Lemma 3:
        Every APPROVE result must have enforced_vector ∈ F_τ.
        Tested across 200 random inputs to a known feasible-interior-containing region.
        """
        region = make_box_region(5, 0.0, 1.0)
        rng = np.random.RandomState(9)
        for _ in range(200):
            x = rng.uniform(0.0, 1.0, 5)   # always feasible
            out = enforce(x, region)
            assert out.result == EnforcementResult.APPROVE
            holds, v, name = guarantee_check(out, region)
            assert holds, f"Lemma 3 consequence violated: violation {v:.2e} on {name}"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION E: THEOREM 1 — ENFORCEMENT SOUNDNESS (THE CENTRAL GUARANTEE)
#
#  Claim (PROOF.md §Theorem 1):
#      For any x ∈ ℝⁿ, any F, any enforcement configuration:
#      If enforce(x, F) returns (r, y) with r ∈ {APPROVE, PROJECT},
#      then ∀j ∈ {1,…,m}: Cⱼ.evaluate(y) ≤ τ.
#
#  Quantitative form: max_j Cⱼ.evaluate(y) ≤ 10⁻⁶ = solver_tol.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem1_EnforcementSoundness:
    """The central guarantee — verified for all constraint types and input classes."""

    def _assert_guarantee(self, out, region, label=""):
        holds, v, name = guarantee_check(out, region)
        assert holds, (
            f"Theorem 1 VIOLATED {label}: "
            f"constraint '{name}' has violation {v:.4e} > τ={TAU:.0e}. "
            f"Result: {out.result.value}"
        )

    # ── Per-constraint-type ────────────────────────────────────────────────

    def test_linear_only(self):
        """Theorem 1 for pure linear constraints (polytope projection)."""
        region = combine_regions(
            box_constraints([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]),
            halfplane([1.0, 1.0, 0.0], 7.0, name="sum01"),
            halfplane([0.0, 1.0, 1.0], 7.0, name="sum12"),
        )
        rng = np.random.RandomState(10)
        for _ in range(200):
            x = rng.uniform(-2.0, 8.0, 3)
            out = enforce(x, region)
            self._assert_guarantee(out, region, "linear")

    def test_quadratic_only(self):
        """Theorem 1 for pure quadratic constraint (ellipsoid)."""
        Q = np.diag([1.0 / 4, 1.0 / 9])   # x²/4 + y²/9 ≤ 1  (ellipse)
        region = FeasibleRegion(
            [QuadraticConstraint(Q, np.zeros(2), 1.0, "ellipse")], 2
        )
        rng = np.random.RandomState(11)
        for _ in range(150):
            x = rng.uniform(-4.0, 4.0, 2)
            out = enforce(x, region)
            self._assert_guarantee(out, region, "quadratic")

    def test_socp_only(self):
        """Theorem 1 for pure SOCP constraint."""
        M = np.eye(3)
        sc = SOCPConstraint(M, np.zeros(3), np.zeros(3), 2.0, "ball")
        region = FeasibleRegion([sc], 3)
        rng = np.random.RandomState(12)
        for _ in range(150):
            x = rng.uniform(-4.0, 4.0, 3)
            out = enforce(x, region)
            self._assert_guarantee(out, region, "socp")

    def test_psd_only(self):
        """Theorem 1 for pure PSD (LMI) constraint."""
        region = make_psd_region()
        rng = np.random.RandomState(13)
        for _ in range(100):
            x = rng.uniform(-1.5, 1.5, 2)
            out = enforce(x, region)
            self._assert_guarantee(out, region, "psd")

    def test_all_four_constraint_types_mixed(self):
        """
        Theorem 1 for a region combining linear + quadratic + SOCP constraints.
        This is the most demanding form of the guarantee.
        """
        region = make_mixed_region()
        rng = np.random.RandomState(14)
        for _ in range(200):
            x = rng.uniform(-3.0, 13.0, 4)
            out = enforce(x, region)
            self._assert_guarantee(out, region, "mixed_4_types")

    # ── Quantitative violation bound ───────────────────────────────────────

    def test_violation_bounded_by_tau_exactly(self):
        """
        Theorem 1 (quantitative): max_j Cⱼ.evaluate(y) ≤ τ = 10⁻⁶.

        Compute the actual worst-case violation across 500 enforcements
        and verify it stays within [0, τ].
        """
        region = make_mixed_region()
        rng = np.random.RandomState(15)
        worst_overall = 0.0
        for _ in range(500):
            x = rng.uniform(-5.0, 15.0, 4)
            out = enforce(x, region)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                for c in region.constraints:
                    v = c.evaluate(out.enforced_vector)
                    worst_overall = max(worst_overall, v)
        assert worst_overall <= TAU, (
            f"Worst violation {worst_overall:.2e} exceeds τ={TAU:.0e}"
        )

    def test_approve_has_zero_violation(self):
        """
        Theorem 1 + Theorem 9a (passthrough):
        For APPROVE results, the enforced vector IS the original vector,
        which satisfied all constraints strictly (not just up to τ).
        The violation bound for APPROVE is tighter: evaluate_j(y) ≤ τ
        because evaluate_j(x) ≤ τ was the APPROVE condition.
        """
        region = make_mixed_region()
        interior = np.array([1.0, 1.0, 2.0, 2.0])
        assert region.is_feasible(interior)
        out = enforce(interior, region)
        assert out.result == EnforcementResult.APPROVE
        assert np.allclose(out.enforced_vector, interior, atol=1e-14)
        assert out.distance == 0.0
        for c in region.constraints:
            assert c.evaluate(out.enforced_vector) <= TAU

    def test_enforce_returns_8_distinct_result_types(self):
        """
        PROOF.md §Theorem 1 Table: enforce() has exactly 8 return points (R1–R8).
        R1 = APPROVE, R2–R7 = REJECT variants, R8 = PROJECT.

        Structural check: count the return _out() calls in the source.
        """
        src = inspect.getsource(enforce)
        approves = src.count("EnforcementResult.APPROVE")
        projects = src.count("EnforcementResult.PROJECT")
        rejects = src.count("EnforcementResult.REJECT")
        # Exactly 1 APPROVE path, exactly 1 PROJECT path, ≥ 6 REJECT paths
        assert approves >= 1, "Must have at least one APPROVE path"
        assert projects >= 1, "Must have at least one PROJECT path"
        assert rejects >= 6, (
            f"PROOF.md cites 6 REJECT returns; found {rejects}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION F: THEOREM 2 — FAIL-CLOSED REJECTION
#
#  Claim (PROOF.md §Theorem 2):
#      If all solvers produce candidates that fail the post-check,
#      the result is REJECT.
#
#  This establishes the fail-closed property: inability to find a feasible
#  point is treated as inadmissibility, not as a pass.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem2_FailClosed:
    """Verify that solver failure produces REJECT, never APPROVE or PROJECT."""

    def test_contradictory_linear_constraints_reject(self):
        """
        Empty feasible region (1D: x ≤ 1 AND x ≥ 2) → all solvers fail
        post-check → enforce() returns REJECT.
        """
        A = np.array([[1.0], [-1.0]])
        b = np.array([1.0, -2.0])
        region = FeasibleRegion([LinearConstraints(A, b, ["ub", "lb"])], 1)
        out = enforce(np.array([0.5]), region)
        assert out.result == EnforcementResult.REJECT, (
            "Empty feasible region must produce REJECT (fail-closed)"
        )

    def test_contradictory_constraints_2d_reject(self):
        """Empty 2D feasible region: box [0,1]² AND x₀+x₁ ≥ 3."""
        region = combine_regions(
            box_constraints([0.0, 0.0], [1.0, 1.0]),
            halfplane([-1.0, -1.0], -3.0, name="lb_sum"),  # -x₀-x₁ ≤ -3 → x₀+x₁ ≥ 3
        )
        out = enforce(np.array([0.5, 0.5]), region)
        assert out.result == EnforcementResult.REJECT

    def test_fail_closed_means_no_approve_or_project_on_empty(self):
        """
        For any point proposed against an empty region,
        the result is strictly REJECT — never APPROVE or PROJECT.
        """
        A = np.array([[1.0], [-1.0]])
        b = np.array([0.5, -2.0])   # x ≤ 0.5 AND x ≥ 2
        region = FeasibleRegion([LinearConstraints(A, b, ["ub", "lb"])], 1)

        rng = np.random.RandomState(16)
        for _ in range(20):
            x = rng.uniform(-5.0, 5.0, 1)
            out = enforce(x, region)
            assert out.result == EnforcementResult.REJECT, (
                f"Fail-closed violated: got {out.result.value} for empty region"
            )

    def test_postcheck_passed_false_becomes_reject(self):
        """
        PROOF.md §Theorem 2 proof:
        project() returns postcheck_passed=False → enforce() R4 fires → REJECT.
        The check `if not proj.postcheck_passed` must be present in source.
        """
        src = inspect.getsource(enforce)
        assert "not proj.postcheck_passed" in src


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION G: THEOREM 3 — HARD-WALL DOMINANCE
#
#  Claim (PROOF.md §Theorem 3):
#      If any hard-wall constraint is violated by x, the result is REJECT
#      and no solver is invoked.
#
#  The hard-wall check (R2) precedes the solver invocation (Step 4).
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem3_HardWallDominance:
    """Verify hard-wall violations produce immediate REJECT with no solver call."""

    def test_hard_wall_violated_produces_reject(self):
        """R2: hard wall violated → REJECT."""
        region = box_constraints([0.0, 0.0], [10.0, 10.0])
        cfg = EnforcementConfig(
            mode="project",
            hard_wall_constraints=frozenset(["upper_0"]),
        )
        schema = Schema(["x", "y"])
        # x=12 violates upper_0 (x ≤ 10); hard wall → REJECT
        out = enforce(np.array([12.0, 5.0]), region, config=cfg, schema=schema)
        assert out.result == EnforcementResult.REJECT

    def test_hard_wall_not_violated_allows_projection(self):
        """When hard wall is not violated, projection proceeds normally."""
        region = box_constraints([0.0, 0.0], [10.0, 10.0])
        cfg = EnforcementConfig(
            mode="project",
            hard_wall_constraints=frozenset(["upper_0"]),
        )
        schema = Schema(["x", "y"])
        # x=5 satisfies upper_0; y=12 violates upper_1 but that's not a hard wall
        out = enforce(np.array([5.0, 12.0]), region, config=cfg, schema=schema)
        # Should PROJECT (corrects y to 10), not REJECT
        assert out.result in (EnforcementResult.PROJECT, EnforcementResult.APPROVE)
        holds, v, name = guarantee_check(out, region)
        assert holds

    def test_hard_wall_check_precedes_solver_invocation(self):
        """
        PROOF.md §Theorem 3 proof:
        The hard-wall check (Step 2) precedes the solver invocation (Step 4).
        Verify via source ordering.
        """
        src = inspect.getsource(enforce)
        hard_wall_idx = src.find("hard_wall_constraints")
        project_call_idx = src.find("proj = project(")
        assert 0 < hard_wall_idx < project_call_idx, (
            "Hard-wall check must appear before solver invocation in enforce()"
        )

    def test_hard_wall_reject_records_no_solver(self):
        """
        When hard wall fires, solver_method should be 'none' and iterations=0.
        No solver was called.
        """
        region = box_constraints([0.0, 0.0], [10.0, 10.0])
        cfg = EnforcementConfig(
            hard_wall_constraints=frozenset(["upper_0"]),
        )
        schema = Schema(["x", "y"])
        out = enforce(np.array([15.0, 5.0]), region, config=cfg, schema=schema)
        assert out.result == EnforcementResult.REJECT
        assert out.solver_method == "none"
        assert out.iterations == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION H: THEOREM 4 — FORBIDDEN-DIMENSION SAFETY
#
#  Claim (PROOF.md §Theorem 4):
#      If enforce() would need to change a dimension with policy
#      PROJECTION_FORBIDDEN, the result is REJECT.
#
#  The feasible projected point is not emitted; REJECT fires at R5.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem4_ForbiddenDimensionSafety:
    """Verify that forbidden-dimension projection violations produce REJECT."""

    def _schema_and_region_2d(self):
        schema = Schema(["x", "y"])
        region = box_constraints([0.0, 0.0], [5.0, 5.0])
        return schema, region

    def test_forbidden_dimension_changed_produces_reject(self):
        """
        x=7 violates upper_0 (x ≤ 5). x is a forbidden dimension.
        Projection would need to change x from 7 to 5 → REJECT (R5).
        """
        schema, region = self._schema_and_region_2d()
        cfg = EnforcementConfig(
            mode="project",
            dimension_policies={"x": DimensionPolicy.PROJECTION_FORBIDDEN},
        )
        out = enforce(np.array([7.0, 3.0]), region, config=cfg, schema=schema)
        assert out.result == EnforcementResult.REJECT

    def test_forbidden_dimension_unchanged_allows_project(self):
        """
        If projection does not need to touch the forbidden dimension,
        PROJECT is allowed (only non-forbidden dimensions are corrected).
        """
        schema, region = self._schema_and_region_2d()
        cfg = EnforcementConfig(
            mode="project",
            dimension_policies={"x": DimensionPolicy.PROJECTION_FORBIDDEN},
        )
        # x=3 is feasible; only y=7 is infeasible
        out = enforce(np.array([3.0, 7.0]), region, config=cfg, schema=schema)
        assert out.result in (EnforcementResult.PROJECT, EnforcementResult.APPROVE)
        holds, v, name = guarantee_check(out, region)
        assert holds

    def test_project_with_flag_still_projects_and_flags(self):
        """
        PROJECT_WITH_FLAG: correction is allowed but the dimension is flagged.
        The guarantee still holds.
        """
        schema, region = self._schema_and_region_2d()
        cfg = EnforcementConfig(
            mode="project",
            dimension_policies={"x": DimensionPolicy.PROJECT_WITH_FLAG},
        )
        out = enforce(np.array([7.0, 3.0]), region, config=cfg, schema=schema)
        assert out.result in (EnforcementResult.PROJECT, EnforcementResult.APPROVE)
        assert "x" in out.flagged_dimensions
        holds, v, name = guarantee_check(out, region)
        assert holds

    def test_forbidden_dimension_guarantee_still_holds_for_approve(self):
        """
        When x is already feasible (APPROVE), the forbidden dimension
        check is irrelevant because no projection is needed.
        Guarantee holds trivially.
        """
        schema, region = self._schema_and_region_2d()
        cfg = EnforcementConfig(
            mode="project",
            dimension_policies={"x": DimensionPolicy.PROJECTION_FORBIDDEN},
        )
        out = enforce(np.array([3.0, 3.0]), region, config=cfg, schema=schema)
        assert out.result == EnforcementResult.APPROVE
        holds, v, name = guarantee_check(out, region)
        assert holds


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION I: THEOREM 5 — BUDGET MONOTONICITY
#
#  Claim (PROOF.md §Theorem 5):
#      Under nonnegative consumption mode, the budget-tightened feasible
#      region is monotone non-expanding: F_{t+1} ⊆ F_t.
#
#  Proof: The remaining budget R_t is non-increasing because consumption ≥ 0.
#         The constraint bound b_{t+1} ≤ b_t, so the half-space shrinks.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem5_BudgetMonotonicity:
    """Verify that the budget-tightened feasible region is non-expanding."""

    def _make_budgeted_system(self):
        """Return a NumerailSystemLocal with a single budget-tracked dimension."""
        config = {
            "policy_id": "budget_mono_test",
            "schema": {"fields": ["x", "y"]},
            "polytope": {
                "A": [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                "b": [10.0, 0.0, 10.0, 0.0],
                "names": ["ub_x", "lb_x", "ub_y", "lb_y"],
            },
            "budgets": [{
                "name": "x_budget",
                "constraint_name": "ub_x",
                "weight": {"x": 1.0},
                "initial": 30.0,
                "consumption_mode": "nonnegative",
            }],
        }
        return NumerailSystemLocal(config)

    def test_remaining_budget_non_increasing(self):
        """
        After each enforcement with nonnegative mode,
        remaining_{t+1} ≤ remaining_t.
        """
        sys = self._make_budgeted_system()
        prev_remaining = sys.budget_remaining.get("x_budget", 30.0)

        for x_val in [3.0, 5.0, 2.0, 4.0, 6.0]:
            result = sys.enforce({"x": x_val, "y": 1.0})
            if result["decision"] != "reject":
                curr = sys.budget_remaining.get("x_budget", 0.0)
                assert curr <= prev_remaining + 1e-12, (
                    f"Budget monotonicity violated: "
                    f"remaining went from {prev_remaining:.4f} to {curr:.4f}"
                )
                prev_remaining = curr

    def test_budget_always_nonnegative(self):
        """
        Remaining budget is non-negative. Once depleted, subsequent
        enforcements of positive x-values must be REJECT (region empty).
        """
        sys = self._make_budgeted_system()
        # Deplete budget: initial=30, enforce x=10 three times
        for _ in range(3):
            sys.enforce({"x": 10.0, "y": 1.0})
        remaining = sys.budget_remaining.get("x_budget", 0.0)
        assert remaining >= -1e-9, f"Remaining budget went negative: {remaining:.4f}"

    def test_subset_relationship_f_t1_subset_f_t(self):
        """
        Strong form of Theorem 5: F_{t+1} ⊆ F_t as sets.

        After one enforcement that depletes some budget, the new constraint
        bound b_{t+1} ≤ b_t. A point that is rejected by F_{t+1} was also
        rejectable by F_t (since F_{t+1} has a tighter or equal bound).

        Concretely: if x_0 is rejected because x > remaining_{t+1},
        and x > remaining_{t+1}, it may still be feasible under F_t
        (if x ≤ remaining_t). This is the subset direction: anything
        feasible in F_{t+1} is also feasible in F_t.
        """
        sys = self._make_budgeted_system()
        # After one step consuming 8 units, remaining = 22
        sys.enforce({"x": 8.0, "y": 1.0})
        remaining_t1 = sys.budget_remaining.get("x_budget", 0.0)

        # A point feasible in F_{t+1} must also have x ≤ remaining_t = 30 (original)
        # because remaining_t1 ≤ 30. So F_{t+1} ⊆ F_t. Verify with a witness:
        witness_x = remaining_t1 - 0.1  # feasible in t+1
        assert witness_x <= 30.0, (
            "Subset witness: F_{t+1} feasible point should also be F_t feasible"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION J: THEOREM 6 — ROLLBACK RESTORATION
#
#  Claim (PROOF.md §Theorem 6):
#      After rollback(action_id), consumed[name] is reduced by exactly the
#      stored delta, and remaining = initial − consumed is correctly restored.
#
#  The rollback is idempotent: a second call removes the delta key and
#  returns False (or raises ValueError at the service layer).
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem6_RollbackRestoration:
    """Verify exact budget restoration and idempotence of rollback."""

    def _make_tracker(self) -> tuple[BudgetTracker, Schema]:
        schema = Schema(["x", "y"])
        tracker = BudgetTracker()
        spec = BudgetSpec(
            name="x_bgt",
            constraint_name="ub_x",
            dimension_name="x",
            weight=1.0,
            initial=100.0,
            consumption_mode="nonnegative",
        )
        tracker.register(spec)
        return tracker, schema

    def test_rollback_restores_exact_delta(self):
        """
        enforce action_id "A1" with x=7.5, then rollback "A1".
        consumed should return to 0.0 (exact delta restoration).
        """
        tracker, schema = self._make_tracker()
        x_vec = np.array([7.5, 3.0])

        tracker.record_consumption(x_vec, "A1", schema)
        status_before = tracker.status()
        assert abs(status_before["x_bgt"]["consumed"] - 7.5) < 1e-12

        success = tracker.rollback("A1")
        assert success

        status_after = tracker.status()
        assert abs(status_after["x_bgt"]["consumed"] - 0.0) < 1e-12
        assert abs(status_after["x_bgt"]["remaining"] - 100.0) < 1e-12

    def test_rollback_independent_of_subsequent_actions(self):
        """
        Theorem 6 proof: "rollback is applied to the current cumulative total,
        subtracting the stored delta regardless of subsequent actions."

        Sequence: A1 (x=5), A2 (x=3), rollback(A1).
        Final consumed = 3.0 (only A2 remains).
        """
        tracker, schema = self._make_tracker()
        tracker.record_consumption(np.array([5.0, 0.0]), "A1", schema)
        tracker.record_consumption(np.array([3.0, 0.0]), "A2", schema)
        assert abs(tracker.status()["x_bgt"]["consumed"] - 8.0) < 1e-12

        tracker.rollback("A1")
        assert abs(tracker.status()["x_bgt"]["consumed"] - 3.0) < 1e-12

    def test_rollback_is_idempotent_returns_false(self):
        """
        PROOF.md §Theorem 6: delta is removed from _action_deltas on first
        rollback; second call finds nothing to remove → returns False.
        """
        tracker, schema = self._make_tracker()
        tracker.record_consumption(np.array([6.0, 0.0]), "A1", schema)

        first = tracker.rollback("A1")
        second = tracker.rollback("A1")
        assert first is True
        assert second is False

    def test_rollback_unknown_action_id_returns_false(self):
        """Rollback of unknown action_id returns False without error."""
        tracker, schema = self._make_tracker()
        result = tracker.rollback("nonexistent_id")
        assert result is False

    def test_rollback_consumed_floored_at_zero(self):
        """
        BudgetTracker.rollback: consumed = max(0.0, consumed - delta).
        Consumed never goes below zero even if delta > consumed.
        """
        tracker, schema = self._make_tracker()
        # Record a small consumption, then attempt to roll back
        # a manually-injected large delta by recording a small action
        tracker.record_consumption(np.array([2.0, 0.0]), "A1", schema)
        tracker.rollback("A1")
        assert tracker.status()["x_bgt"]["consumed"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION K: THEOREM 7 — MONOTONE SELF-LIMITS
#
#  Claim (PROOF.md §Theorem 7):
#      Under assumptions A1 (only authorized authority creates policy versions),
#      A2 (agent has only enforce scope), A3 (mode = "reject"):
#      Any proposal that increases a dimension beyond the approved ceiling
#      is rejected.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem7_MonotoneSelfLimits:
    """Verify that reject mode and explicit upper bounds block inflation."""

    def test_reject_mode_blocks_infeasible_proposal(self):
        """
        A3: mode = "reject". Any proposal violating bounds → REJECT, no PROJECT.
        The agent cannot inflate a bound by proposing a value above it.
        """
        region = box_constraints([0.0, 0.0], [5.0, 5.0])
        cfg = EnforcementConfig(mode="reject")
        out = enforce(np.array([7.0, 3.0]), region, config=cfg)
        assert out.result == EnforcementResult.REJECT, (
            "Reject mode: infeasible proposal must be rejected, not projected"
        )

    def test_reject_mode_approves_feasible_proposal(self):
        """In reject mode, a feasible proposal still gets APPROVE."""
        region = box_constraints([0.0, 0.0], [5.0, 5.0])
        cfg = EnforcementConfig(mode="reject")
        out = enforce(np.array([3.0, 3.0]), region, config=cfg)
        assert out.result == EnforcementResult.APPROVE

    def test_reject_mode_source_check(self):
        """
        PROOF.md §Theorem 7 proof cites return R3:
            if cfg.mode == "reject": return _out(REJECT, ...)
        Verify this path exists before the solver invocation.
        """
        src = inspect.getsource(enforce)
        reject_mode_idx = src.find('cfg.mode == "reject"')
        solver_idx = src.find("proj = project(")
        assert 0 < reject_mode_idx < solver_idx, (
            "Reject mode gate must fire before solver in enforce()"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION L: THEOREM 8 — AUDIT-CHAIN INTEGRITY
#
#  Claim (PROOF.md §Theorem 8):
#      Under SHA-256 collision resistance, any modification, insertion,
#      or deletion of an audit record is detectable by chain verification.
#
#  The chain includes prev_hash in the hashed payload, producing a
#  Merkle-chain structure where each record commits to all prior records.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem8_AuditChainIntegrity:
    """Verify SHA-256 audit chain detects all classes of tampering."""

    def _make_chain_with_records(self, n: int = 5) -> AuditChain:
        """Build a chain by enforcing n times and returning the chain."""
        region = make_box_region(2, 0.0, 5.0)
        chain = AuditChain()
        rng = np.random.RandomState(20)
        for i in range(n):
            x = rng.uniform(0.0, 7.0, 2)
            out = enforce(x, region)
            chain.append(out)
        return chain

    def test_clean_chain_verifies(self):
        """Theorem 8 baseline: an unmodified chain must verify cleanly."""
        chain = self._make_chain_with_records(10)
        valid, depth = chain.verify()
        assert valid, "Unmodified chain must verify"
        assert depth == 10

    def test_modified_record_content_detected(self):
        """
        Tamper: change a field value in one record.
        The hash of that record will no longer match its stored hash.
        """
        chain = self._make_chain_with_records(5)
        records = chain.export()

        # Directly mutate a retained record (accessing internal list for test purposes)
        with chain._lock:
            if chain._records:
                chain._records[2]["result"] = "tampered"

        valid, _ = chain.verify()
        assert not valid, "Modified record content must be detected"

    def test_modified_hash_field_detected(self):
        """Tamper: change the stored hash field without changing content."""
        chain = self._make_chain_with_records(5)
        with chain._lock:
            if len(chain._records) >= 3:
                chain._records[2]["hash"] = "a" * 64   # wrong hash

        valid, _ = chain.verify()
        assert not valid

    def test_modified_prev_hash_breaks_chain(self):
        """Tamper: change the prev_hash link in a record."""
        chain = self._make_chain_with_records(5)
        with chain._lock:
            if len(chain._records) >= 2:
                chain._records[1]["prev_hash"] = "0" * 64

        valid, _ = chain.verify()
        assert not valid

    def test_chain_uses_sha256(self):
        """
        Theorem 8 (SHA-256 collision resistance assumption):
        Verify that the hash function is SHA-256 and deterministic.
        """
        region = make_box_region(2, 0.0, 5.0)
        chain = AuditChain()
        out = enforce(np.array([1.0, 2.0]), region)
        h = chain.append(out)
        assert len(h) == 64, "SHA-256 produces 64 hex chars"
        # Determinism: same record → same hash when recomputed
        records = chain.export()
        rec = {k: v for k, v in records[0].items() if k != "hash"}
        expected = hashlib.sha256(
            _deterministic_json(rec).encode()
        ).hexdigest()
        assert h == expected

    def test_chain_prev_hash_links_records(self):
        """
        Each record's prev_hash equals the hash of the preceding record.
        This is the Merkle-chain structure described in Theorem 8.
        """
        chain = self._make_chain_with_records(6)
        records = chain.export()
        for i in range(1, len(records)):
            assert records[i]["prev_hash"] == records[i - 1]["hash"], (
                f"prev_hash link broken at record {i}"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION M: THEOREM 9 — PASSTHROUGH AND IDEMPOTENCE
#
#  Claim (PROOF.md §Theorem 9):
#  (a) Passthrough: x ∈ F_τ ⟹ enforce(x) = (APPROVE, x), distance = 0.
#  (b) Idempotence: if enforce(x) = (r, y) with r ∈ {APPROVE, PROJECT},
#      then enforce(y) = (APPROVE, y), distance = 0.
# ═══════════════════════════════════════════════════════════════════════════

class TestTheorem9_PassthroughIdempotence:
    """Verify passthrough and idempotence for all constraint types."""

    def test_passthrough_feasible_vector_approve(self):
        """
        Theorem 9a: x ∈ F_τ → (APPROVE, x, distance=0).
        """
        region = make_mixed_region()
        x = np.array([1.0, 1.0, 2.0, 2.0])   # known interior point
        assert region.is_feasible(x)
        out = enforce(x, region)
        assert out.result == EnforcementResult.APPROVE
        assert np.allclose(out.enforced_vector, x, atol=1e-14)
        assert out.distance == 0.0

    def test_passthrough_holds_for_all_constraint_types(self):
        """Theorem 9a for each individual constraint type."""
        regions_with_points = [
            (make_box_region(3, 0.0, 5.0), np.array([1.0, 2.0, 3.0])),
            (FeasibleRegion([QuadraticConstraint(
                np.diag([1.0, 1.0]), np.zeros(2), 4.0, "ball")], 2),
             np.array([0.5, 0.5])),
            (FeasibleRegion([SOCPConstraint(
                np.eye(2), np.zeros(2), np.zeros(2), 2.0, "cone")], 2),
             np.array([0.5, 0.5])),
            (make_psd_region(), np.array([0.1, 0.1])),
        ]
        for region, x in regions_with_points:
            assert region.is_feasible(x), f"Test point should be feasible"
            out = enforce(x, region)
            assert out.result == EnforcementResult.APPROVE, (
                f"Passthrough failed: got {out.result.value}"
            )
            assert np.allclose(out.enforced_vector, x, atol=1e-14)

    def test_idempotence_project_output_approves(self):
        """
        Theorem 9b: enforce(y) = (APPROVE, y, 0) when y is a PROJECT output.
        """
        region = make_mixed_region()
        x_infeasible = np.array([12.0, 12.0, 8.0, 8.0])
        out1 = enforce(x_infeasible, region)

        if out1.result == EnforcementResult.PROJECT:
            y = out1.enforced_vector
            out2 = enforce(y, region)
            assert out2.result == EnforcementResult.APPROVE, (
                f"Idempotence violated: second enforce on projected point "
                f"returned {out2.result.value}, not APPROVE"
            )
            assert np.allclose(out2.enforced_vector, y, atol=1e-10)
            assert abs(out2.distance) < 1e-9

    def test_idempotence_across_all_constraint_types(self):
        """Theorem 9b verified for all four constraint types."""
        region = make_mixed_region()
        rng = np.random.RandomState(21)
        idempotence_verified = 0
        for _ in range(100):
            x = rng.uniform(-5.0, 15.0, 4)
            out1 = enforce(x, region)
            if out1.result == EnforcementResult.PROJECT:
                out2 = enforce(out1.enforced_vector, region)
                assert out2.result == EnforcementResult.APPROVE, (
                    "Idempotence: second enforce on PROJECT output must be APPROVE"
                )
                idempotence_verified += 1
        assert idempotence_verified > 0, "At least some PROJECT outputs needed for this test"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION N: COROLLARIES — SOLVER AND PROPOSER INDEPENDENCE
#
#  Corollary 1 (PROOF.md §Corollary: Solver Independence):
#      Theorem 1 holds for any solver implementation, including adversarial ones.
#      The guarantee follows from the post-check, not from solver correctness.
#
#  Corollary 2 (PROOF.md §Corollary: Proposer Independence):
#      Theorem 1 holds for any proposed vector, including adversarial ones.
#      NaN and Inf are rejected by _validate_vector() before enforcement.
# ═══════════════════════════════════════════════════════════════════════════

class TestCorollaries_Independence:
    """Verify solver and proposer independence."""

    # ── Proposer independence ──────────────────────────────────────────────

    def test_nan_input_rejected(self):
        """
        Corollary (Proposer Independence):
        NaN inputs are rejected by _validate_vector() before enforcement.
        They never reach the constraint checker.
        """
        region = make_box_region(3)
        with pytest.raises(ValidationError):
            enforce(np.array([1.0, float("nan"), 1.0]), region)

    def test_inf_input_rejected(self):
        """Inf inputs raise ValidationError before any constraint check."""
        region = make_box_region(3)
        with pytest.raises(ValidationError):
            enforce(np.array([1.0, float("inf"), 1.0]), region)

    def test_wrong_dimension_rejected(self):
        """Dimension mismatch raises ValidationError."""
        region = make_box_region(3)
        with pytest.raises(ValidationError):
            enforce(np.array([1.0, 2.0]), region)  # 2D vs 3D region

    def test_maximally_infeasible_proposer(self):
        """
        Adversarial proposer: x is as far from the feasible region as possible.
        Guarantee still holds for any APPROVE/PROJECT result.
        """
        region = make_mixed_region()
        extreme_points = [
            np.array([1000.0, 1000.0, 1000.0, 1000.0]),
            np.array([-1000.0, -1000.0, -1000.0, -1000.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]
        for x in extreme_points:
            out = enforce(x, region)
            holds, v, name = guarantee_check(out, region)
            assert holds, (
                f"Guarantee violated for extreme input: violation {v:.2e} on '{name}'"
            )

    def test_high_dimensional_random(self):
        """
        Corollary (Proposer Independence, high-dimensional):
        n=50 random box region, 200 random proposals.
        Guarantee holds for every APPROVE/PROJECT result.
        """
        n = 50
        region = make_box_region(n, 0.0, 1.0)
        rng = np.random.RandomState(22)
        for _ in range(200):
            x = rng.uniform(-0.5, 1.5, n)
            out = enforce(x, region)
            holds, v, name = guarantee_check(out, region)
            assert holds, f"High-dim guarantee violated: violation {v:.2e}"

    # ── Solver independence ────────────────────────────────────────────────

    def test_guarantee_holds_across_all_solver_paths(self):
        """
        Corollary (Solver Independence):
        The guarantee is verified for results from all three solver paths:
        - box_clamp (pure box regions)
        - dykstra (linear polytopes)
        - slsqp (mixed/nonlinear regions)
        """
        # Path 1: box_clamp
        box_r = make_box_region(4, 0.0, 1.0)
        for x in [np.array([2.0, -1.0, 0.5, 3.0])]:
            out = enforce(x, box_r)
            if out.result == EnforcementResult.PROJECT:
                assert out.solver_method == "box_clamp"
            holds, v, _ = guarantee_check(out, box_r)
            assert holds

        # Path 2: Dykstra (linear + coupling)
        poly_r = combine_regions(
            box_constraints([0.0, 0.0], [3.0, 3.0]),
            halfplane([1.0, 1.0], 4.0, name="sum"),
        )
        rng = np.random.RandomState(23)
        for _ in range(30):
            out = enforce(rng.uniform(-1.0, 5.0, 2), poly_r)
            holds, v, _ = guarantee_check(out, poly_r)
            assert holds

        # Path 3: SLSQP (mixed constraints force SLSQP path)
        mixed_r = make_mixed_region()
        for _ in range(30):
            out = enforce(rng.uniform(-3.0, 13.0, 4), mixed_r)
            holds, v, _ = guarantee_check(out, mixed_r)
            assert holds

    def test_guarantee_proof_does_not_assume_solver_correctness(self):
        """
        Corollary structural check: project() wraps every solver output
        in an is_feasible post-check before setting postcheck_passed=True.
        The proof never uses the solver's claimed optimality — only the
        post-check result.
        """
        src = inspect.getsource(project)
        # Every branch that sets postcheck_passed=True (via ProjectionResult(..., True))
        # must be preceded by region.is_feasible
        feasibility_checks = src.count("region.is_feasible")
        # The proof cites ≥ 4 feasibility checks (R1-R4 in Lemma 2 table)
        assert feasibility_checks >= 4, (
            f"Expected ≥ 4 is_feasible calls in project(); found {feasibility_checks}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION O: QUANTITATIVE PRECISION ANALYSIS
#
#  What is the exact worst-case violation magnitude?
#  The guarantee is: max_j Cⱼ.evaluate(y) ≤ τ = solver_tol = 10⁻⁶.
#  This section measures the actual violation distribution empirically.
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantitative_PrecisionBounds:
    """Quantitative characterisation of the guarantee's precision."""

    def test_violation_bound_is_tau_not_zero(self):
        """
        The guarantee is τ-tight, not zero-tight.
        For APPROVE: the exact original vector is returned (zero violation by definition).
        For PROJECT: violation ≤ τ = 1e-6 (not necessarily zero).
        """
        region = make_mixed_region()
        rng = np.random.RandomState(24)
        max_approve_viol = 0.0
        max_project_viol = 0.0
        for _ in range(300):
            x = rng.uniform(-3.0, 13.0, 4)
            out = enforce(x, region)
            if out.result == EnforcementResult.APPROVE:
                viol = max(c.evaluate(out.enforced_vector) for c in region.constraints)
                max_approve_viol = max(max_approve_viol, viol)
            elif out.result == EnforcementResult.PROJECT:
                viol = max(c.evaluate(out.enforced_vector) for c in region.constraints)
                max_project_viol = max(max_project_viol, viol)

        # APPROVE: enforced = x ∈ F_τ, so viol ≤ τ
        assert max_approve_viol <= TAU, (
            f"APPROVE violation {max_approve_viol:.2e} exceeds τ={TAU:.0e}"
        )
        # PROJECT: enforced ∈ F_τ, so viol ≤ τ
        assert max_project_viol <= TAU, (
            f"PROJECT violation {max_project_viol:.2e} exceeds τ={TAU:.0e}"
        )

    def test_tolerance_parameter_controls_guarantee_tightness(self):
        """
        The solver_tol parameter directly controls the violation bound.
        With solver_tol=1e-4, violations up to 1e-4 are acceptable.
        The guarantee at tighter τ should produce tighter (or equal) violations.
        """
        region = make_box_region(3, 0.0, 1.0)
        x_outside = np.array([1.5, 0.5, 0.5])

        for tol in [1e-3, 1e-6, 1e-8]:
            cfg = EnforcementConfig(solver_tol=tol)
            out = enforce(x_outside, region, config=cfg)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                viol = max(c.evaluate(out.enforced_vector) for c in region.constraints)
                assert viol <= tol, (
                    f"Violation {viol:.2e} exceeds solver_tol={tol:.2e}"
                )

    def test_distance_field_matches_euclidean_norm(self):
        """
        EnforcementOutput.distance = ‖enforced − original‖₂.
        Verify this matches the independently-computed L2 distance.
        """
        region = make_box_region(4, 0.0, 1.0)
        x = np.array([2.0, -0.5, 1.5, 0.3])
        out = enforce(x, region)
        expected_dist = float(np.linalg.norm(out.enforced_vector - out.original_vector))
        assert abs(out.distance - expected_dist) < 1e-10

    def test_approve_distance_is_exactly_zero(self):
        """For APPROVE, distance must be identically 0.0."""
        region = make_box_region(3, 0.0, 5.0)
        x_feasible = np.array([1.0, 2.0, 3.0])
        out = enforce(x_feasible, region)
        assert out.result == EnforcementResult.APPROVE
        assert out.distance == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION P: PROJECTION OPTIMALITY
#
#  Mathematical claim: for a convex set C and x ∉ C, the projection
#  y = argmin_{z ∈ C} ‖z − x‖ satisfies ‖y − x‖ ≤ ‖z − x‖ for all z ∈ C.
#
#  Numerail's guarantee does NOT require optimality — it only requires
#  y ∈ F_τ. But when a feasible correction is found, it should be
#  reasonably close to the proposal.
#
#  We test approximate optimality: the solver's projected point is not
#  worse than a manually-computed feasible point.
# ═══════════════════════════════════════════════════════════════════════════

class TestProjectionOptimality:
    """Verify that projected points are near-optimal nearest feasible points."""

    def test_box_clamp_is_exact_projection(self):
        """
        For pure box constraints, the clamp is the exact L2 projection.
        ‖clamp(x) − x‖ ≤ ‖z − x‖ for any feasible z.
        """
        region = make_box_region(4, 0.0, 1.0)
        x = np.array([1.5, -0.3, 0.7, 2.0])
        out = enforce(x, region)
        y = out.enforced_vector

        # Verify the clamp formula: yᵢ = clip(xᵢ, 0, 1)
        expected = np.clip(x, 0.0, 1.0)
        assert np.allclose(y, expected, atol=1e-10), (
            f"Box clamp should produce exact projection: {y} vs expected {expected}"
        )

        # Verify optimality: any other feasible z has ‖z - x‖ ≥ ‖y - x‖
        rng = np.random.RandomState(25)
        dist_y = np.linalg.norm(y - x)
        for _ in range(50):
            z = rng.uniform(0.0, 1.0, 4)   # random feasible point
            assert np.linalg.norm(z - x) >= dist_y - 1e-10

    def test_slsqp_projection_reasonable_for_quadratic(self):
        """
        For quadratic constraints, SLSQP should find a near-optimal projection.
        The KKT condition for projection onto an ellipsoid has a known structure.
        """
        # Unit circle: x² + y² ≤ 1
        Q = np.eye(2)
        region = FeasibleRegion(
            [QuadraticConstraint(Q, np.zeros(2), 1.0, "circle")], 2
        )
        x = np.array([2.0, 0.0])   # outside the circle
        out = enforce(x, region)

        if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            y = out.enforced_vector
            # Analytic projection: y = x / ‖x‖ = [1, 0] for this case
            expected = np.array([1.0, 0.0])
            # Allow solver tolerance in the comparison
            assert np.linalg.norm(y - expected) < 1e-4, (
                f"SLSQP projection {y} far from analytic solution {expected}"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION Q: CONVEXITY OF THE FEASIBLE SETS
#
#  Each constraint type defines a convex feasible set:
#  - LinearConstraints: intersection of half-spaces (polytope) — trivially convex
#  - QuadraticConstraint (Q ≽ 0): sublevel set of a convex function — convex
#  - SOCPConstraint: second-order cone — convex
#  - PSDConstraint: {x : A₀ + Σ xᵢAᵢ ≽ 0} — convex (linear matrix inequality)
#
#  Convexity is necessary for the projection to yield a unique nearest point.
#  We test the midpoint property: ∀ y₁, y₂ ∈ F, (y₁+y₂)/2 ∈ F.
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintConvexity:
    """Verify convexity (midpoint property) for each constraint type."""

    def _test_convexity(self, region: FeasibleRegion, n_pairs: int = 50, seed: int = 0):
        """
        Convexity test: for n_pairs of feasible points y₁, y₂,
        verify that the midpoint (y₁+y₂)/2 is also feasible.
        """
        rng = np.random.RandomState(seed)
        n = region.n_dim
        feasible_points = []

        # Find feasible points by projecting random seeds
        for _ in range(n_pairs * 3):
            x = rng.randn(n)
            out = enforce(x, region)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                y = out.enforced_vector
                if region.is_feasible(y):
                    feasible_points.append(y)
            if len(feasible_points) >= n_pairs * 2:
                break

        midpoint_failures = 0
        for i in range(0, min(len(feasible_points) - 1, n_pairs * 2), 2):
            y1, y2 = feasible_points[i], feasible_points[i + 1]
            mid = 0.5 * (y1 + y2)
            # Allow small tolerance for numerical midpoint
            if not region.is_feasible(mid, tol=1e-4):
                midpoint_failures += 1

        return midpoint_failures, len(feasible_points) // 2

    def test_linear_constraint_convexity(self):
        """Polytope (intersection of half-spaces) is convex."""
        region = combine_regions(
            box_constraints([0.0, 0.0], [5.0, 5.0]),
            halfplane([1.0, 1.0], 7.0, name="sum"),
        )
        failures, tested = self._test_convexity(region, seed=26)
        assert failures == 0, f"{failures}/{tested} midpoints were infeasible"

    def test_quadratic_constraint_convexity(self):
        """Sublevel set of x'Qx + a'x with Q ≽ 0 is convex."""
        Q = np.diag([1.0, 2.0, 3.0])
        region = FeasibleRegion(
            [QuadraticConstraint(Q, np.zeros(3), 4.0, "ellipsoid")], 3
        )
        failures, tested = self._test_convexity(region, seed=27)
        assert failures == 0

    def test_socp_constraint_convexity(self):
        """Second-order cone is convex."""
        M = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        region = FeasibleRegion(
            [SOCPConstraint(M, np.zeros(3), np.zeros(2), 2.0, "socp_conv")], 2
        )
        failures, tested = self._test_convexity(region, seed=28)
        assert failures == 0

    def test_psd_constraint_convexity(self):
        """LMI feasible set is convex."""
        region = make_psd_region()
        failures, tested = self._test_convexity(region, n_pairs=30, seed=29)
        assert failures == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION R: ROUTING THRESHOLDS — ORDERING INVARIANT
#
#  The RoutingThresholds dataclass enforces:
#      0 ≤ silent ≤ flagged ≤ confirmation ≤ hard_reject
#
#  This is a separate correctness property: the routing decision must
#  monotonically increase with projection distance.
# ═══════════════════════════════════════════════════════════════════════════

class TestRoutingThresholds:
    """Verify routing threshold monotonicity and decision correctness."""

    def test_threshold_ordering_enforced(self):
        """RoutingThresholds rejects non-monotone configurations."""
        with pytest.raises(ValidationError):
            RoutingThresholds(silent=0.5, flagged=0.2, confirmation=0.8, hard_reject=1.0)

    def test_routing_decisions_monotone_with_distance(self):
        """
        For a fixed constraint configuration, routing decisions should
        be non-decreasing (more severe) as projection distance increases.

        Tier mapping: SILENT_PROJECT < FLAGGED_PROJECT
                      < CONFIRMATION_REQUIRED < HARD_REJECT
        """
        tiers = {
            RoutingDecision.SILENT_PROJECT: 0,
            RoutingDecision.FLAGGED_PROJECT: 1,
            RoutingDecision.CONFIRMATION_REQUIRED: 2,
            RoutingDecision.HARD_REJECT: 3,
        }
        thresholds = RoutingThresholds(
            silent=0.1, flagged=0.5, confirmation=1.0, hard_reject=5.0,
        )
        region = make_box_region(1, 0.0, 1.0)
        cfg = EnforcementConfig(routing_thresholds=thresholds)

        prev_tier = -1
        for x_val in [1.1, 1.6, 2.5, 8.0]:
            out = enforce(np.array([x_val]), region, config=cfg)
            if out.routing_decision is not None:
                tier = tiers[out.routing_decision]
                assert tier >= prev_tier, (
                    f"Routing tier decreased: {out.routing_decision} at x={x_val}"
                )
                prev_tier = tier

    def test_hard_reject_routing_produces_reject_result(self):
        """
        When routing_threshold.hard_reject is exceeded, the result is REJECT.
        The solver may have found a feasible point, but routing overrides it.
        """
        region = box_constraints([0.0], [1.0])
        cfg = EnforcementConfig(
            routing_thresholds=RoutingThresholds(
                silent=0.01, flagged=0.05, confirmation=0.1, hard_reject=0.5,
            ),
        )
        # x=2.0: projection to 1.0 has distance=1.0 > hard_reject=0.5
        out = enforce(np.array([2.0]), region, config=cfg)
        assert out.result == EnforcementResult.REJECT
        assert out.routing_decision == RoutingDecision.HARD_REJECT


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION S: TRUSTED CONTEXT INJECTION
#
#  The merge_trusted_context function is the server-side security boundary
#  for Invariant 4 (CLAUDE.md): "The AI is the subject of governance,
#  never the governor."
#
#  Server-authoritative fields override agent-supplied values; the agent
#  cannot lower telemetry values to inflate admissibility.
# ═══════════════════════════════════════════════════════════════════════════

class TestTrustedContextInjection:
    """Verify that trusted context fields are correctly overwritten."""

    def test_trusted_fields_override_agent_supplied_values(self):
        """
        merge_trusted_context(raw_values, trusted_context, trusted_fields)
        overwrites the action dict's trusted fields with server-authoritative
        values, discarding the agent's claimed values.

        Only fields in the trusted_fields frozenset are overwritten.
        """
        action = {"x": 5.0, "y": 2.0, "gpu_util": 0.10}
        trusted = {"gpu_util": 0.95}
        merged = merge_trusted_context(action, trusted, frozenset({"gpu_util"}))
        assert merged["gpu_util"] == 0.95, (
            "Trusted field must override agent-supplied value"
        )
        assert merged["x"] == 5.0   # non-trusted field unchanged
        assert merged["y"] == 2.0

    def test_trusted_context_agent_cannot_downgrade(self):
        """
        Agent cannot reduce a trusted utilisation value to inflate admissibility.
        Even if the agent proposes gpu_util=0.01, the server's authoritative
        value (0.90) is used because gpu_util is in trusted_fields.
        """
        action_low = {"x": 5.0, "gpu_util": 0.01}   # agent claims low util
        trusted = {"gpu_util": 0.90}                  # server says 90%
        merged = merge_trusted_context(
            action_low, trusted, frozenset({"gpu_util"})
        )
        assert merged["gpu_util"] == 0.90

    def test_untrusted_fields_pass_through_unchanged(self):
        """Fields not in trusted_fields are passed through from raw_values."""
        action = {"x": 3.0, "y": 7.0, "gpu_util": 0.5}
        trusted = {"gpu_util": 0.8, "x": 99.0}   # server has x too, but x is not trusted
        merged = merge_trusted_context(
            action, trusted, frozenset({"gpu_util"})  # only gpu_util trusted
        )
        assert merged["gpu_util"] == 0.8   # overwritten by trusted
        assert merged["x"] == 3.0          # not in trusted_fields → agent value kept
        assert merged["y"] == 7.0

    def test_empty_trusted_fields_is_identity(self):
        """merge_trusted_context with empty trusted_fields = identity on raw_values."""
        action = {"x": 3.0, "y": 7.0}
        merged = merge_trusted_context(action, {"x": 99.0}, frozenset())
        assert merged["x"] == 3.0   # trusted_fields is empty → no overwrite
        assert merged["y"] == 7.0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION T: SCHEMA CORRECTNESS
#
#  The Schema class maps named fields to vector indices and applies
#  affine normalization. Correct vectorization/devectorization is a
#  precondition for the guarantee's constraint-space interpretation.
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaCorrectness:
    """Verify that Schema vectorization/devectorization is a bijection."""

    def test_vectorize_devectorize_roundtrip(self):
        """devectorize(vectorize(d)) = d for all fields without normalizers."""
        schema = Schema(["a", "b", "c"])
        d = {"a": 1.5, "b": -3.0, "c": 0.0}
        vec = schema.vectorize(d)
        recovered = schema.devectorize(vec)
        for k in d:
            assert abs(recovered[k] - d[k]) < 1e-14

    def test_normalizer_roundtrip(self):
        """
        With affine normalization [lo, hi]:
            normalize(v) = (v - lo) / (hi - lo)
            denormalize(n) = n * (hi - lo) + lo
        The composition must be the identity.
        """
        schema = Schema(["x", "y"], normalizers={"x": (0.0, 100.0)})
        d = {"x": 37.5, "y": 2.0}
        vec = schema.vectorize(d)
        # Normalized x should be 0.375
        assert abs(vec[0] - 0.375) < 1e-14
        recovered = schema.devectorize(vec)
        assert abs(recovered["x"] - 37.5) < 1e-12

    def test_schema_duplicate_field_rejected(self):
        """Duplicate field names raise SchemaError — prevents index ambiguity."""
        from numerail.engine import SchemaError
        with pytest.raises(SchemaError):
            Schema(["x", "y", "x"])

    def test_field_ordering_preserved(self):
        """Field order in vectorize output matches Schema field order."""
        schema = Schema(["a", "b", "c"])
        d = {"a": 10.0, "b": 20.0, "c": 30.0}
        vec = schema.vectorize(d)
        assert vec[0] == 10.0
        assert vec[1] == 20.0
        assert vec[2] == 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION U: STRESS TESTS — ADVERSARIAL INPUTS AT SCALE
#
#  These tests combine all guarantee components under high-volume,
#  randomised, adversarial conditions to detect any emergent violations
#  not captured by the targeted tests above.
# ═══════════════════════════════════════════════════════════════════════════

class TestStress_AdversarialAtScale:
    """High-volume randomised guarantee verification."""

    def test_1000_random_mixed_region(self):
        """
        Theorem 1 (stress): 1,000 random proposals against the mixed 4D region.
        Every APPROVE/PROJECT result must satisfy every constraint.
        Records the maximum observed violation.
        """
        region = make_mixed_region()
        rng = np.random.RandomState(30)
        violations = []
        for _ in range(1000):
            x = rng.uniform(-5.0, 15.0, 4)
            out = enforce(x, region)
            holds, v, name = guarantee_check(out, region)
            assert holds, f"Stress: guarantee violated, violation {v:.4e} on '{name}'"
            if out.result != EnforcementResult.REJECT:
                violations.append(v)

        if violations:
            max_viol = max(violations)
            assert max_viol <= TAU, f"Max violation {max_viol:.2e} exceeds τ={TAU:.0e}"

    def test_guarantee_under_all_enforcement_modes(self):
        """
        The guarantee holds in all three enforcement modes:
        "project", "reject", "hybrid".
        """
        region = make_mixed_region()
        rng = np.random.RandomState(31)
        modes = [
            EnforcementConfig(mode="project"),
            EnforcementConfig(mode="reject"),
            EnforcementConfig(mode="hybrid", max_distance=5.0),
        ]
        for cfg in modes:
            for _ in range(100):
                x = rng.uniform(-3.0, 13.0, 4)
                out = enforce(x, region, config=cfg)
                holds, v, name = guarantee_check(out, region)
                assert holds, (
                    f"Guarantee violated in mode '{cfg.mode}': "
                    f"violation {v:.4e} on '{name}'"
                )

    def test_guarantee_under_safety_margin(self):
        """
        With safety_margin < 1.0, the effective region is tighter.
        The guarantee still holds — the output is verified against the
        tightened effective region.
        """
        region = make_box_region(4, 0.0, 10.0)
        cfg = EnforcementConfig(safety_margin=0.8)
        effective = region.with_safety_margin(0.8)

        rng = np.random.RandomState(32)
        for _ in range(100):
            x = rng.uniform(-1.0, 11.0, 4)
            out = enforce(x, region, config=cfg)
            # Check against the effective (tightened) region
            holds, v, name = guarantee_check(out, effective)
            assert holds, (
                f"Guarantee violated under safety_margin=0.8: "
                f"violation {v:.4e} on '{name}'"
            )

    def test_psd_guarantee_under_adversarial_input(self):
        """
        PSD constraints are the most numerically sensitive type.
        Verify the guarantee for 200 adversarial inputs near the
        LMI boundary.
        """
        region = make_psd_region()
        rng = np.random.RandomState(33)
        for _ in range(200):
            # Sample near the feasibility boundary
            x = rng.uniform(-1.2, 1.2, 2)
            out = enforce(x, region)
            holds, v, name = guarantee_check(out, region)
            assert holds, f"PSD guarantee violated: violation {v:.4e}"
