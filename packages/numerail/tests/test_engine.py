"""Engine-level tests: constraints, enforce, budgets, audit, rollback."""

import numpy as np
import pytest

from numerail.engine import (
    enforce, project, box_constraints, halfplane, combine_regions, ellipsoid,
    check_feasibility, chebyshev_center, synthesize_feedback,
    NumerailSystem, Schema, BudgetSpec, AuditChain, MetricsCollector,
    RegionVersionStore, FeasibleRegion,
    ConvexConstraint, LinearConstraints, QuadraticConstraint,
    SOCPConstraint, PSDConstraint,
    EnforcementResult, EnforcementConfig, EnforcementOutput, RollbackResult,
    DimensionPolicy, RoutingDecision, RoutingThresholds,
    NumerailError, ValidationError, ConstraintError,
    InfeasibleRegionError, SolverError, SchemaError, ResolutionError,
    Polytope, ActionSchema, GCESystem, GCEError,
)


# ── Linear constraints ──────────────────────────────────────────────────

class TestLinearConstraints:
    def test_box_approve(self):
        r = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([0.5, 0.5]), r)
        assert out.result == EnforcementResult.APPROVE
        np.testing.assert_array_almost_equal(out.enforced_vector, [0.5, 0.5])

    def test_box_project(self):
        r = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([1.5, 0.5]), r)
        assert out.result == EnforcementResult.PROJECT
        assert r.is_feasible(out.enforced_vector)

    def test_box_project_negative(self):
        r = box_constraints([0, 0], [1, 1])
        out = enforce(np.array([-0.5, 2.0]), r)
        assert out.result == EnforcementResult.PROJECT
        assert r.is_feasible(out.enforced_vector)

    def test_halfplane(self):
        hp = halfplane([1, 1], 1.0, name="sum")
        r = combine_regions(box_constraints([0, 0], [1, 1]), hp)
        out = enforce(np.array([0.8, 0.8]), r)
        assert out.result == EnforcementResult.PROJECT
        assert r.is_feasible(out.enforced_vector)

    def test_named_constraints(self):
        r = box_constraints([0, 0], [1, 1], names=["ux", "uy", "lx", "ly"])
        out = enforce(np.array([1.5, 0.5]), r)
        assert "ux" in out.violated_constraints


# ── Quadratic constraints ────────────────────────────────────────────────

class TestQuadraticConstraints:
    def test_sphere(self):
        q = QuadraticConstraint(np.eye(2), np.zeros(2), 1.0, "sphere")
        r = FeasibleRegion([q], 2)
        out = enforce(np.array([0.5, 0.5]), r)
        assert out.result == EnforcementResult.APPROVE

    def test_sphere_project(self):
        q = QuadraticConstraint(np.eye(2), np.zeros(2), 1.0, "sphere")
        r = FeasibleRegion([q], 2)
        out = enforce(np.array([2.0, 2.0]), r)
        assert out.result == EnforcementResult.PROJECT
        assert r.is_feasible(out.enforced_vector)

    def test_non_psd_rejected(self):
        with pytest.raises(ConstraintError):
            QuadraticConstraint(np.array([[-1, 0], [0, 1]]), np.zeros(2), 1.0)


# ── SOCP constraints ────────────────────────────────────────────────────

class TestSOCPConstraints:
    def test_socp_feasible(self):
        s = SOCPConstraint(np.eye(2), np.zeros(2), np.array([0.0, 0.0]), 2.0, "cone")
        r = FeasibleRegion([s], 2)
        out = enforce(np.array([0.5, 0.5]), r)
        assert out.result == EnforcementResult.APPROVE

    def test_socp_project(self):
        s = SOCPConstraint(np.eye(2), np.zeros(2), np.array([0.0, 0.0]), 1.0, "cone")
        r = FeasibleRegion([s], 2)
        out = enforce(np.array([2.0, 2.0]), r)
        assert out.result in (EnforcementResult.PROJECT, EnforcementResult.REJECT)
        if out.result == EnforcementResult.PROJECT:
            assert r.is_feasible(out.enforced_vector)


# ── PSD constraints ──────────────────────────────────────────────────────

class TestPSDConstraints:
    def test_psd_feasible(self):
        A0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        A1 = np.array([[-1.0, 0.0], [0.0, 0.0]])
        p = PSDConstraint(A0, [A1], "psd")
        r = FeasibleRegion([p], 1)
        out = enforce(np.array([0.5]), r)
        assert out.result == EnforcementResult.APPROVE


# ── Mixed constraints ────────────────────────────────────────────────────

class TestMixedConstraints:
    def test_linear_plus_quadratic(self):
        lin = LinearConstraints(
            np.vstack([np.eye(2), -np.eye(2)]),
            np.array([2, 2, 0, 0], dtype=float),
            names=["ux", "uy", "lx", "ly"],
        )
        quad = QuadraticConstraint(np.eye(2), np.zeros(2), 1.0, "ball")
        m = FeasibleRegion([lin, quad], 2)
        out = enforce(np.array([1.5, 1.5]), m)
        assert m.is_feasible(out.enforced_vector)


# ── Enforcement modes ────────────────────────────────────────────────────

class TestEnforcementModes:
    def test_reject_mode(self):
        r = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(mode="reject")
        out = enforce(np.array([1.5, 0.5]), r, cfg)
        assert out.result == EnforcementResult.REJECT

    def test_hybrid_mode_within_distance(self):
        r = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(mode="hybrid", max_distance=1.0)
        out = enforce(np.array([1.1, 0.5]), r, cfg)
        assert out.result == EnforcementResult.PROJECT

    def test_hybrid_mode_exceeds_distance(self):
        r = box_constraints([0, 0], [1, 1])
        cfg = EnforcementConfig(mode="hybrid", max_distance=0.01)
        out = enforce(np.array([1.5, 0.5]), r, cfg)
        assert out.result == EnforcementResult.REJECT

    def test_hard_wall(self):
        r = box_constraints([0, 0], [1, 1], names=["ux", "uy", "lx", "ly"])
        cfg = EnforcementConfig(hard_wall_constraints=frozenset({"ux"}))
        out = enforce(np.array([1.5, 0.5]), r, cfg)
        assert out.result == EnforcementResult.REJECT

    def test_forbidden_dimension(self):
        r = box_constraints([0, 0], [1, 1])
        sch = Schema(fields=["x", "y"])
        cfg = EnforcementConfig(dimension_policies={"x": DimensionPolicy.PROJECTION_FORBIDDEN})
        out = enforce(np.array([1.5, 1.5]), r, cfg, sch)
        assert out.result == EnforcementResult.REJECT


# ── Validation ───────────────────────────────────────────────────────────

class TestValidation:
    def test_nan_raises(self):
        r = box_constraints([0], [1])
        with pytest.raises(ValidationError):
            enforce(np.array([float("nan")]), r)

    def test_inf_raises(self):
        r = box_constraints([0], [1])
        with pytest.raises(ValidationError):
            enforce(np.array([float("inf")]), r)

    def test_dimension_mismatch(self):
        r = box_constraints([0, 0], [1, 1])
        with pytest.raises(ValidationError):
            enforce(np.array([0.5]), r)


# ── Schema ───────────────────────────────────────────────────────────────

class TestSchema:
    def test_vectorize_devectorize(self):
        sch = Schema(fields=["x", "y", "z"])
        v = sch.vectorize({"x": 1.0, "y": 2.0, "z": 3.0})
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
        d = sch.devectorize(v)
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_defaults(self):
        sch = Schema(fields=["x", "y"], defaults={"y": 0.5})
        v = sch.vectorize({"x": 1.0})
        np.testing.assert_array_equal(v, [1.0, 0.5])

    def test_normalizers(self):
        sch = Schema(fields=["x"], normalizers={"x": (0.0, 10.0)})
        v = sch.vectorize({"x": 5.0})
        assert 0.0 <= v[0] <= 1.0


# ── Budget + rollback ────────────────────────────────────────────────────

class TestBudgetAndRollback:
    def _make_system(self):
        sch = Schema(fields=["amount"])
        reg = combine_regions(
            box_constraints([0], [100], names=["max_amount", "min_amount"]),
            halfplane([1], 100.0, name="cap"),
        )
        sys = NumerailSystem(sch, reg)
        sys.register_budget(BudgetSpec(
            name="spend", constraint_name="cap",
            dimension_name="amount", weight=1.0, initial=100.0,
        ))
        return sys

    def test_budget_consumption(self):
        sys = self._make_system()
        sys.enforce({"amount": 30.0}, action_id="a1")
        status = sys.budget_status()
        assert status["spend"]["consumed"] == pytest.approx(30.0)

    def test_rollback_result_type(self):
        sys = self._make_system()
        sys.enforce({"amount": 30.0}, action_id="a1")
        rb = sys.rollback("a1")
        assert isinstance(rb, RollbackResult)
        assert rb.rolled_back is True
        assert bool(rb) is True

    def test_rollback_restores_budget(self):
        sys = self._make_system()
        sys.enforce({"amount": 30.0}, action_id="a1")
        sys.rollback("a1")
        status = sys.budget_status()
        assert status["spend"]["consumed"] == pytest.approx(0.0)

    def test_double_rollback_returns_false(self):
        sys = self._make_system()
        sys.enforce({"amount": 30.0}, action_id="a1")
        rb1 = sys.rollback("a1")
        rb2 = sys.rollback("a1")
        assert rb1.rolled_back is True
        assert rb2.rolled_back is False
        assert bool(rb2) is False

    def test_weight_map_budget(self):
        sch = Schema(fields=["a", "b"])
        reg = combine_regions(
            box_constraints([0, 0], [100, 100], names=["ma", "mb", "la", "lb"]),
            halfplane([1, 1], 200.0, name="total"),
        )
        sys = NumerailSystem(sch, reg)
        sys.register_budget(BudgetSpec(
            name="combo", constraint_name="total",
            initial=200.0, weight_map={"a": 0.6, "b": 0.4},
        ))
        sys.enforce({"a": 10.0, "b": 20.0}, action_id="a1")
        status = sys.budget_status()
        expected = 10.0 * 0.6 + 20.0 * 0.4  # 14.0
        assert status["combo"]["consumed"] == pytest.approx(expected)


# ── Audit chain ──────────────────────────────────────────────────────────

class TestAuditChain:
    def test_append_and_verify(self):
        chain = AuditChain(max_records=100)
        r = box_constraints([0, 0], [1, 1])
        out1 = enforce(np.array([0.5, 0.5]), r)
        out2 = enforce(np.array([1.5, 0.5]), r)
        h1 = chain.append(out1)
        h2 = chain.append(out2)
        assert h1 != h2
        valid, depth = chain.verify()
        assert valid
        assert depth == 2

    def test_tamper_detection(self):
        chain = AuditChain(max_records=100)
        r = box_constraints([0, 0], [1, 1])
        chain.append(enforce(np.array([0.5, 0.5]), r))
        chain.append(enforce(np.array([1.5, 0.5]), r))
        # Tamper with the first record
        if chain._records:
            chain._records[0]["result"] = "tampered"
        valid, _ = chain.verify()
        assert not valid


# ── Backward compatibility ───────────────────────────────────────────────

class TestBackwardCompat:
    def test_polytope_wrapper(self):
        p = Polytope(np.eye(2), np.array([1.0, 1.0]))
        assert p.n_dimensions == 2
        assert p.n_constraints == 2
        assert p.contains(np.array([0.5, 0.5]))
        assert not p.contains(np.array([1.5, 0.5]))

    def test_aliases(self):
        assert ActionSchema is Schema
        assert GCESystem is NumerailSystem
        assert GCEError is NumerailError


# ── Guarantee fuzz ───────────────────────────────────────────────────────

class TestGuaranteeFuzz:
    def test_box_fuzz(self):
        rng = np.random.RandomState(42)
        r = box_constraints([0, 0, 0], [1, 1, 1])
        for _ in range(100):
            x = rng.uniform(-2, 3, size=3)
            out = enforce(x, r)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                assert r.is_feasible(out.enforced_vector)

    def test_mixed_fuzz(self):
        rng = np.random.RandomState(99)
        reg = combine_regions(
            box_constraints([0, 0, 0], [1, 1, 1]),
            halfplane([1, 1, 1], 2.0, name="sum"),
        )
        for _ in range(100):
            x = rng.uniform(-2, 3, size=3)
            out = enforce(x, reg)
            if out.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                assert reg.is_feasible(out.enforced_vector)

    def test_idempotence(self):
        rng = np.random.RandomState(7)
        r = box_constraints([0, 0], [1, 1])
        for _ in range(50):
            x = rng.uniform(-1, 2, size=2)
            out1 = enforce(x, r)
            if out1.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                out2 = enforce(out1.enforced_vector, r)
                assert out2.result == EnforcementResult.APPROVE


# ── Projection routing regression ────────────────────────────────────


class TestProjectionRouting:
    """Regression for linear-only mixed regions: Dykstra should be preferred
    over SLSQP for polytopes that are not pure axis-aligned boxes."""

    def test_linear_only_mixed_projection_prefers_dykstra(self):
        reg = combine_regions(
            box_constraints([0, 0, 0], [1, 1, 1]),
            halfplane([1, 1, 1], 2.0, name="sum"),
        )
        # Known sample where SLSQP hits iteration limit (~0.95s)
        # but Dykstra solves in 2 iterations (~0.0001s)
        x = np.array([-1.96587133, 1.84896514, 1.7338355])
        proj = project(x, reg)
        assert proj.postcheck_passed
        assert reg.is_feasible(proj.point)
        assert proj.solver_method in ("dykstra", "box_clamp")

    def test_nonlinear_mixed_still_uses_slsqp_first(self):
        """Quadratic + linear region should still try SLSQP before Dykstra."""
        lin = LinearConstraints(
            np.vstack([np.eye(2), -np.eye(2)]),
            np.array([2, 2, 0, 0], dtype=float),
            names=["ux", "uy", "lx", "ly"],
        )
        quad = QuadraticConstraint(np.eye(2), np.zeros(2), 1.0, "ball")
        reg = FeasibleRegion([lin, quad], 2)
        x = np.array([1.5, 1.5])
        proj = project(x, reg)
        assert proj.postcheck_passed
        assert reg.is_feasible(proj.point)
        # SLSQP should handle this cleanly
        assert proj.solver_method == "slsqp"
