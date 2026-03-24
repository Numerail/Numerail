"""Numerail v5.0.0 — Enforcement Guarantee Certification Suite.

Independently verifies the central mathematical guarantee:

    r ∈ {APPROVE, PROJECT} ⟹ ∀ c ∈ F_t.constraints : c.evaluate(y) ≤ τ

7 categories, 45 tests. Run: pytest tests/test_guarantee.py -v
"""

import inspect

import numpy as np
import pytest

import numerail as nm
from numerail.engine import (
    enforce, project, box_constraints, halfplane, combine_regions,
    FeasibleRegion, LinearConstraints, QuadraticConstraint,
    SOCPConstraint, PSDConstraint,
    EnforcementResult, EnforcementConfig, EnforcementOutput,
    DimensionPolicy, RoutingDecision, RoutingThresholds,
    NumerailSystem, Schema, BudgetSpec, AuditChain,
    ValidationError, ConstraintError, SchemaError, ResolutionError,
    merge_trusted_context,
)


def check_guarantee(out, region, tol=1e-6):
    """Check the guarantee for a single enforcement result.
    Returns (holds, worst_violation)."""
    if out.result not in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
        return True, 0.0
    worst = 0.0
    for c in region.constraints:
        v = c.evaluate(out.enforced_vector)
        if v > worst:
            worst = v
    return worst <= tol, worst


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 1: STRUCTURAL VERIFICATION
# ═════════════════════════════════════════════════════════════════════════

class TestStructural:
    """Verifies that every APPROVE and PROJECT code path in enforce()
    is gated by a feasibility post-check, with no bypass possible."""

    def test_approve_path_gated_by_is_feasible(self):
        source = inspect.getsource(enforce)
        assert "effective.is_feasible(x)" in source

    def test_project_path_gated_by_postcheck(self):
        source = inspect.getsource(enforce)
        assert "proj.postcheck_passed" in source
        assert "not proj.postcheck_passed" in source

    def test_postcheck_set_only_by_is_feasible(self):
        source = inspect.getsource(project)
        assert source.count("region.is_feasible") >= 4

    def test_emit_path_assertion(self):
        source = inspect.getsource(enforce)
        # Check for the defense-in-depth check (explicit raise, not assert)
        assert "effective.is_feasible(enforced" in source
        assert "AssertionError" in source

    def test_reject_returns_never_approved(self):
        source = inspect.getsource(enforce)
        lines = source.split("\n")
        for line in lines:
            if "EnforcementResult.REJECT" in line and "return" in line.lower():
                assert "APPROVE" not in line
                assert "PROJECT" not in line


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 2: FORMAL PROPERTY TESTS (mirror the 9 theorems)
# ═════════════════════════════════════════════════════════════════════════

class TestFormalProperties:
    """Mirrors the 9 theorems as executable property tests."""

    def test_theorem1_soundness_random(self):
        region = combine_regions(
            box_constraints([0, 0, 0], [1, 1, 1]),
            halfplane([1, 1, 1], 2.0, name="sum"),
        )
        rng = np.random.RandomState(42)
        for _ in range(100):
            x = rng.uniform(-2, 3, size=3)
            out = enforce(x, region)
            holds, v = check_guarantee(out, region)
            assert holds, f"Violation {v:.2e} at x={x}"

    def test_theorem1b_passthrough(self):
        region = box_constraints([0, 0], [1, 1])
        x = np.array([0.5, 0.5])
        out = enforce(x, region)
        assert out.result == EnforcementResult.APPROVE
        np.testing.assert_array_almost_equal(out.enforced_vector, x)
        assert out.distance == 0.0

    def test_theorem1c_idempotence(self):
        region = box_constraints([0, 0], [1, 1])
        out1 = enforce(np.array([2.0, 0.5]), region)
        out2 = enforce(out1.enforced_vector, region)
        assert out1.result == EnforcementResult.PROJECT
        assert out2.result == EnforcementResult.APPROVE
        assert out2.distance == 0.0

    def test_theorem2_fail_closed(self):
        region = box_constraints([0, 0], [0.001, 0.001])
        out = enforce(np.array([1000.0, 1000.0]), region)
        if out.result == EnforcementResult.PROJECT:
            assert region.is_feasible(out.enforced_vector)

    def test_theorem3_hard_wall_dominance(self):
        region = combine_regions(
            box_constraints([0, 0], [1, 1], names=["mx", "my", "lx", "ly"]),
            halfplane([1, 1], 1.5, name="budget"),
        )
        cfg = EnforcementConfig(hard_wall_constraints=frozenset({"budget"}))
        out = enforce(np.array([1.0, 1.0]), region, cfg)
        assert out.result == EnforcementResult.REJECT
        assert out.solver_method == "none"

    def test_theorem4_forbidden_dimension(self):
        schema = Schema(fields=["x", "y"])
        cfg = EnforcementConfig(
            dimension_policies={"x": DimensionPolicy.PROJECTION_FORBIDDEN},
        )
        out = enforce(np.array([1.5, 0.5]), box_constraints([0, 0], [1, 1]), cfg, schema)
        assert out.result == EnforcementResult.REJECT

    def test_theorem5_budget_monotonicity(self):
        schema = Schema(fields=["cost", "qty"])
        region = combine_regions(
            box_constraints([-1, 0], [1, 1], names=["mc", "mq", "lc", "lq"]),
            halfplane([1, 0], 1.0, name="cap"),
        )
        sys = NumerailSystem(schema, region)
        sys.register_budget(BudgetSpec(
            name="b", constraint_name="cap", dimension_name="cost",
            weight=1.0, initial=1.0, consumption_mode="nonnegative",
        ))
        sys.enforce({"cost": -0.5, "qty": 0.5}, action_id="a1")
        assert sys.budget_status()["b"]["consumed"] == 0.0

    def test_theorem6_rollback_restoration(self):
        schema = Schema(fields=["cost", "qty"])
        region = combine_regions(
            box_constraints([0, 0], [0.5, 1], names=["mc", "mq", "lc", "lq"]),
            halfplane([1, 0], 0.5, name="cap"),
        )
        sys = NumerailSystem(schema, region)
        sys.register_budget(BudgetSpec(
            name="b", constraint_name="cap", dimension_name="cost",
            weight=1.0, initial=0.5,
        ))
        sys.enforce({"cost": 0.3, "qty": 0.5}, action_id="a1")
        sys.enforce({"cost": 0.15, "qty": 0.5}, action_id="a2")
        pre = sys.budget_status()["b"]["remaining"]
        sys.rollback("a2")
        post = sys.budget_status()["b"]["remaining"]
        assert abs(pre - 0.05) < 1e-4
        assert abs(post - 0.20) < 1e-4

    def test_theorem8_audit_integrity(self):
        schema = Schema(fields=["x", "y"])
        region = box_constraints([0, 0], [1, 1])
        sys = NumerailSystem(schema, region)
        rng = np.random.RandomState(123)
        for i in range(100):
            sys.enforce({"x": rng.uniform(-1, 2), "y": rng.uniform(-1, 2)}, action_id=f"a{i}")
        valid, depth = sys.verify_audit()
        assert valid
        assert depth == 100

    def test_theorem9_passthrough_and_idempotence(self):
        region = box_constraints([0, 0], [1, 1])
        rng = np.random.RandomState(7)
        for _ in range(50):
            x = rng.uniform(-1, 2, size=2)
            out1 = enforce(x, region)
            if out1.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                out2 = enforce(out1.enforced_vector, region)
                assert out2.result == EnforcementResult.APPROVE

    def test_composed_safety_3_authorities(self):
        auth1 = box_constraints([0, 0, 0], [1, 1, 1], tag="ops")
        auth2 = FeasibleRegion([
            LinearConstraints(np.array([[1, 1, 0]]), np.array([1.2]),
                              names=["budget"], tags=["finance"]),
        ], 3)
        auth3 = FeasibleRegion([
            LinearConstraints(np.array([[0, 1, 1]]), np.array([1.0]),
                              names=["risk_cap"], tags=["risk"]),
        ], 3)
        combined = FeasibleRegion.combine(auth1, auth2, auth3)
        out = enforce(np.array([0.9, 0.8, 0.7]), combined)
        if out.result == EnforcementResult.PROJECT:
            assert auth1.is_feasible(out.enforced_vector)
            assert auth2.is_feasible(out.enforced_vector)
            assert auth3.is_feasible(out.enforced_vector)


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 3: CONSTRAINT TYPE COVERAGE
# ═════════════════════════════════════════════════════════════════════════

class TestConstraintTypes:
    """Verifies the guarantee holds for each constraint type and compositions."""

    def test_linear_project(self):
        region = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([1.5, 0.5]), region)
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_quadratic_project(self):
        quad = QuadraticConstraint(np.eye(3), np.zeros(3), 1.0, "ball")
        region = FeasibleRegion([quad], 3)
        out = enforce(np.array([2.0, 2.0, 2.0]), region)
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_socp_project(self):
        socp = SOCPConstraint(np.eye(3), np.zeros(3), np.zeros(3), 2.0, "norm2")
        region = FeasibleRegion([socp], 3)
        out = enforce(np.array([3.0, 4.0, 0.0]), region)
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_psd_project(self):
        psd = PSDConstraint(np.eye(2), [np.eye(2), np.array([[0, 1.0], [1, 0]])], "lmi")
        region = FeasibleRegion([psd], 2)
        out = enforce(np.array([-2.0, 0.0]), region)
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_mixed_linear_quadratic(self):
        lin = LinearConstraints(
            np.vstack([np.eye(2), -np.eye(2)]),
            np.array([2, 2, 0, 0], dtype=float),
            names=["ux", "uy", "lx", "ly"],
        )
        quad = QuadraticConstraint(np.eye(2), np.zeros(2), 1.0, "ball")
        region = FeasibleRegion([lin, quad], 2)
        out = enforce(np.array([1.5, 1.5]), region)
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_mixed_fuzz_100(self):
        lin = LinearConstraints(
            np.vstack([np.eye(3), -np.eye(3)]),
            np.concatenate([2 * np.ones(3), np.zeros(3)]),
            names=[f"u{i}" for i in range(3)] + [f"l{i}" for i in range(3)],
        )
        quad = QuadraticConstraint(np.eye(3), np.zeros(3), 1.0, "ball3d")
        region = FeasibleRegion([lin, quad], 3)
        rng = np.random.RandomState(42)
        for _ in range(100):
            x = rng.randn(3) * 3
            out = enforce(x, region)
            holds, v = check_guarantee(out, region)
            assert holds, f"Violation {v:.2e}"


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 4: ADVERSARIAL PROBES
# ═════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    """Tests edge cases, attack vectors, and potential weaknesses."""

    def test_nan_rejection(self):
        with pytest.raises(ValidationError):
            enforce(np.array([np.nan, 0.5]), box_constraints([0, 0], [1, 1]))

    def test_inf_rejection(self):
        with pytest.raises(ValidationError):
            enforce(np.array([np.inf, 0.5]), box_constraints([0, 0], [1, 1]))

    def test_dimension_mismatch(self):
        with pytest.raises(ValidationError):
            enforce(np.array([0.5]), box_constraints([0, 0], [1, 1]))

    def test_no_output_aliasing(self):
        region = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([1.5, 0.5]), region)
        out.enforced_vector[0] = 999
        out2 = enforce(np.array([0.5, 0.5]), region)
        np.testing.assert_array_almost_equal(out2.enforced_vector, [0.5, 0.5])

    def test_safety_margin_subset(self):
        region = box_constraints([-2, 0.5], [3, 2])
        tight = region.with_safety_margin(0.8)
        rng = np.random.RandomState(0)
        for _ in range(1000):
            x = rng.uniform(-3, 4, size=2)
            if tight.is_feasible(x):
                assert region.is_feasible(x)

    def test_bugfix_safety_margin_negative_bounds(self):
        region = box_constraints([0.2, 0.2], [1.0, 1.0])
        tight = region.with_safety_margin(0.8)
        assert not tight.is_feasible(np.array([0.22, 0.5]))
        assert tight.is_feasible(np.array([0.5, 0.5]))

    def test_bugfix_audit_eviction(self):
        chain = AuditChain(max_records=3)
        for _ in range(5):
            out = enforce(np.array([0.5, 0.5, 0.5]), box_constraints([0, 0, 0], [1, 1, 1]))
            chain.append(out)
        valid, depth = chain.verify()
        assert valid
        assert depth == 3

    def test_duplicate_name_detection(self):
        with pytest.raises(ConstraintError):
            LinearConstraints(np.eye(2), np.ones(2), names=["dup", "dup"])

    def test_non_psd_rejection(self):
        with pytest.raises(ConstraintError):
            QuadraticConstraint(np.array([[-1, 0], [0, 1]]), np.zeros(2), 1.0)

    def test_missing_hard_wall_name(self):
        with pytest.raises(ResolutionError):
            enforce(
                np.array([1.5, 0.5]),
                box_constraints([0, 0], [1, 1]),
                EnforcementConfig(hard_wall_constraints=frozenset({"nonexistent"})),
            )

    def test_unknown_dimension_policy_key(self):
        sch = Schema(fields=["x", "y"])
        cfg = EnforcementConfig(
            dimension_policies={"z_bad": DimensionPolicy.PROJECTION_FORBIDDEN},
        )
        with pytest.raises(SchemaError):
            enforce(np.array([1.5, 0.5]), box_constraints([0, 0], [1, 1]), cfg, sch)

    def test_trusted_context_merge(self):
        raw = {"amount": 100.0, "risk": 0.1}
        trusted = {"risk": 0.85}
        merged = merge_trusted_context(raw, trusted, frozenset({"risk"}))
        assert merged["amount"] == 100.0
        assert merged["risk"] == 0.85

    def test_from_config_bad_dim_policy(self):
        with pytest.raises(SchemaError):
            NumerailSystem.from_config({
                "schema": {"fields": ["x", "y"]},
                "polytope": {
                    "A": [[1, 0], [0, 1], [-1, 0], [0, -1]], "b": [1, 1, 0, 0],
                    "names": ["ux", "uy", "lx", "ly"],
                },
                "enforcement": {
                    "mode": "project",
                    "dimension_policies": {"nonexistent": "forbidden"},
                },
            })


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 5: RANDOMIZED STRESS TEST
# ═════════════════════════════════════════════════════════════════════════

class TestStress:
    """Runs the guarantee check across random constraint compositions."""

    def test_random_compositions(self):
        rng = np.random.RandomState(777)
        worst_violation = 0.0
        for trial in range(100):
            n = rng.randint(2, 5)
            lo = rng.uniform(-1, 0, size=n)
            hi = rng.uniform(0.5, 2, size=n)
            A = np.vstack([np.eye(n), -np.eye(n)])
            b = np.concatenate([hi, -lo])
            names = [f"u{i}" for i in range(n)] + [f"l{i}" for i in range(n)]
            constraints = [LinearConstraints(A, b, names=names)]
            if rng.random() < 0.15:
                Q = np.eye(n) * rng.uniform(0.5, 1.5)
                constraints.append(
                    QuadraticConstraint(Q, np.zeros(n), rng.uniform(1, 3), f"q{trial}")
                )
            region = FeasibleRegion(constraints, n)
            for _ in range(3):
                x = rng.uniform(-3, 4, size=n)
                out = enforce(x, region)
                holds, v = check_guarantee(out, region)
                assert holds, f"Trial {trial}: violation {v:.2e}"
                if v > worst_violation:
                    worst_violation = v
        assert worst_violation <= 1e-6

    def test_high_dimensional_fuzz(self):
        rng = np.random.RandomState(555)
        for n in [5, 10, 20]:
            lo = rng.uniform(-1, 0, size=n)
            hi = rng.uniform(0.5, 2, size=n)
            region = box_constraints(lo, hi)
            for _ in range(20):
                x = rng.uniform(-3, 4, size=n)
                out = enforce(x, region)
                holds, v = check_guarantee(out, region)
                assert holds, f"dim={n}: violation {v:.2e}"


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 6: ENFORCEMENT MODE COVERAGE
# ═════════════════════════════════════════════════════════════════════════

class TestModes:
    """Verifies the guarantee under all three enforcement modes."""

    def test_mode_project(self):
        region = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([1.5, 0.5]), region, EnforcementConfig(mode="project"))
        assert out.result == EnforcementResult.PROJECT
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_mode_reject(self):
        region = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([1.5, 0.5]), region, EnforcementConfig(mode="reject"))
        assert out.result == EnforcementResult.REJECT

    def test_mode_hybrid_within(self):
        region = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(mode="hybrid", max_distance=0.6)
        out = enforce(np.array([1.5, 0.5]), region, cfg)
        assert out.result == EnforcementResult.PROJECT
        holds, _ = check_guarantee(out, region)
        assert holds

    def test_mode_hybrid_exceeds(self):
        region = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(mode="hybrid", max_distance=0.1)
        out = enforce(np.array([5.0, 5.0]), region, cfg)
        assert out.result == EnforcementResult.REJECT

    def test_routing_thresholds(self):
        region = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(
            routing_thresholds=RoutingThresholds(
                silent=0.05, flagged=0.2, confirmation=0.5, hard_reject=1.0,
            ),
        )
        out = enforce(np.array([1.02, 0.5]), region, cfg)
        if out.result == EnforcementResult.PROJECT:
            assert out.routing_decision == RoutingDecision.SILENT_PROJECT


# ═════════════════════════════════════════════════════════════════════════
#  SECTION 7: TOLERANCE PRECISION
# ═════════════════════════════════════════════════════════════════════════

class TestTolerance:
    """Quantifies the precision of the guarantee under different solver_tol."""

    @pytest.mark.parametrize("tol", [1e-9, 1e-6, 1e-4])
    def test_tolerance_precision(self, tol):
        region = combine_regions(
            box_constraints([0, 0, 0], [1, 1, 1]),
            halfplane([1, 1, 1], 2.0, name="sum"),
        )
        rng = np.random.RandomState(42)
        cfg = EnforcementConfig(solver_tol=tol)
        worst = 0.0
        for _ in range(50):
            x = rng.uniform(-2, 3, size=3)
            out = enforce(x, region, cfg)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                for c in region.constraints:
                    v = c.evaluate(out.enforced_vector)
                    if v > worst:
                        worst = v
        assert worst <= tol, f"worst={worst:.2e} exceeds tol={tol:.0e}"
