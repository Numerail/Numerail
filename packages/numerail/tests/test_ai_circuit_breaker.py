"""Tests for the AI circuit breaker control-plane reserve pattern.

Verifies that the reserve-aware feasible region construction causes
Numerail's existing guarantee to imply control-plane survivability:
any approved or projected action preserves the encoded controller reserve.

Tests the production-shaped path through NumerailSystemLocal.
"""

import numpy as np
import pytest

from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_config():
    """Build the circuit breaker policy config."""
    fields = [
        "prompt_k", "completion_k", "tool_calls", "external_api_calls",
        "gpu_seconds", "parallel_workers", "current_gpu_util", "current_api_util",
        "ctrl_gpu_reserve_seconds", "ctrl_api_reserve_calls", "ctrl_parallel_reserve",
        "gpu_disturbance_margin_seconds", "api_disturbance_margin_calls",
    ]
    n = len(fields)
    idx = {f: i for i, f in enumerate(fields)}

    def row(coeffs):
        r = [0.0] * n
        for f, v in coeffs.items():
            r[idx[f]] = v
        return r

    maxes = {
        "prompt_k": 64, "completion_k": 16, "tool_calls": 40,
        "external_api_calls": 20, "gpu_seconds": 120, "parallel_workers": 16,
        "current_gpu_util": 1, "current_api_util": 1,
        "ctrl_gpu_reserve_seconds": 40, "ctrl_api_reserve_calls": 6,
        "ctrl_parallel_reserve": 2,
        "gpu_disturbance_margin_seconds": 20, "api_disturbance_margin_calls": 4,
    }

    A_rows, b_rows, names = [], [], []
    for f in fields:
        A_rows.append(row({f: 1.0})); b_rows.append(float(maxes[f])); names.append(f"max_{f}")
    for f in fields:
        A_rows.append(row({f: -1.0})); b_rows.append(0.0); names.append(f"min_{f}")

    A_rows.append(row({"external_api_calls": 1, "tool_calls": -1}))
    b_rows.append(0.0); names.append("external_le_tool_calls")

    A_rows.append(row({"current_gpu_util": 1, "gpu_seconds": 1/240,
                        "ctrl_gpu_reserve_seconds": 1/240, "gpu_disturbance_margin_seconds": 1/240}))
    b_rows.append(0.90); names.append("gpu_headroom_reserve")

    A_rows.append(row({"current_api_util": 1, "external_api_calls": 1/40,
                        "ctrl_api_reserve_calls": 1/40, "api_disturbance_margin_calls": 1/40}))
    b_rows.append(0.90); names.append("api_headroom_reserve")

    A_rows.append(row({"parallel_workers": 1, "ctrl_parallel_reserve": 1}))
    b_rows.append(16.0); names.append("parallel_headroom_reserve")

    A_rows.append(row({"gpu_seconds": 1})); b_rows.append(600.0); names.append("remaining_gpu_day")
    A_rows.append(row({"external_api_calls": 1})); b_rows.append(80.0); names.append("remaining_api_day")

    Q_diag = [1/64**2, 1/16**2, 1/40**2, 1/20**2, 1/120**2, 1/16**2] + [0]*7

    M_socp = np.zeros((3, n))
    M_socp[0, idx["current_gpu_util"]] = 1.0; M_socp[0, idx["gpu_seconds"]] = 1/240
    M_socp[0, idx["ctrl_gpu_reserve_seconds"]] = 1/240; M_socp[0, idx["gpu_disturbance_margin_seconds"]] = 1/240
    M_socp[1, idx["current_api_util"]] = 1.0; M_socp[1, idx["external_api_calls"]] = 1/40
    M_socp[1, idx["ctrl_api_reserve_calls"]] = 1/40; M_socp[1, idx["api_disturbance_margin_calls"]] = 1/40
    M_socp[2, idx["parallel_workers"]] = 1/16; M_socp[2, idx["ctrl_parallel_reserve"]] = 1/16

    A0_psd = np.eye(2).tolist()
    A_psd = [[[0,0],[0,0]] for _ in range(n)]
    A_psd[idx["gpu_seconds"]] = [[-1/240,0],[0,0]]
    A_psd[idx["ctrl_gpu_reserve_seconds"]] = [[-1/240,0],[0,0]]
    A_psd[idx["gpu_disturbance_margin_seconds"]] = [[-1/240,0],[0,0]]
    A_psd[idx["current_gpu_util"]] = [[-1,0],[0,0]]
    A_psd[idx["external_api_calls"]] = [[0,0],[0,-1/40]]
    A_psd[idx["ctrl_api_reserve_calls"]] = [[0,0],[0,-1/40]]
    A_psd[idx["api_disturbance_margin_calls"]] = [[0,0],[0,-1/40]]
    A_psd[idx["current_api_util"]] = [[0,0],[0,-1]]
    A_psd[idx["parallel_workers"]] = [[0,0.35/16],[0.35/16,0]]
    A_psd[idx["ctrl_parallel_reserve"]] = [[0,0.35/16],[0.35/16,0]]

    return {
        "policy_id": "ai_circuit_breaker_v1",
        "schema": {"fields": fields},
        "polytope": {"A": [list(r) for r in A_rows], "b": b_rows, "names": names},
        "quadratic_constraints": [{"Q": np.diag(Q_diag).tolist(), "a": [0]*n, "b": 2.25, "name": "resource_energy"}],
        "socp_constraints": [{"M": M_socp.tolist(), "q": [0]*3, "c": [0]*n, "d": 1.15, "name": "burst_envelope"}],
        "psd_constraints": [{"A0": A0_psd, "A_list": A_psd, "name": "coupled_headroom_psd"}],
        "trusted_fields": ["current_gpu_util","current_api_util","ctrl_gpu_reserve_seconds",
                           "ctrl_api_reserve_calls","ctrl_parallel_reserve",
                           "gpu_disturbance_margin_seconds","api_disturbance_margin_calls"],
        "enforcement": {
            "mode": "project",
            "dimension_policies": {
                "current_gpu_util": "forbidden", "current_api_util": "forbidden",
                "ctrl_gpu_reserve_seconds": "forbidden", "ctrl_api_reserve_calls": "forbidden",
                "ctrl_parallel_reserve": "forbidden",
                "gpu_disturbance_margin_seconds": "forbidden", "api_disturbance_margin_calls": "forbidden",
                "parallel_workers": "project_with_flag",
            },
            "routing_thresholds": {"silent": 0.05, "flagged": 1.5, "confirmation": 4.0, "hard_reject": 8.0},
        },
        "budgets": [
            {"name": "gpu_day", "constraint_name": "remaining_gpu_day",
             "weight": {"gpu_seconds": 1.0}, "initial": 600.0, "consumption_mode": "nonnegative"},
            {"name": "api_day", "constraint_name": "remaining_api_day",
             "weight": {"external_api_calls": 1.0}, "initial": 80.0, "consumption_mode": "nonnegative"},
        ],
    }


def _default_trusted():
    return {
        "current_gpu_util": 0.30, "current_api_util": 0.20,
        "ctrl_gpu_reserve_seconds": 30, "ctrl_api_reserve_calls": 4,
        "ctrl_parallel_reserve": 2,
        "gpu_disturbance_margin_seconds": 15, "api_disturbance_margin_calls": 3,
    }


def _light_trusted():
    return {
        "current_gpu_util": 0.20, "current_api_util": 0.20,
        "ctrl_gpu_reserve_seconds": 15, "ctrl_api_reserve_calls": 2,
        "ctrl_parallel_reserve": 1,
        "gpu_disturbance_margin_seconds": 8, "api_disturbance_margin_calls": 2,
    }


def _safe_workload():
    return {
        "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
        "external_api_calls": 5, "gpu_seconds": 60, "parallel_workers": 4,
    }


def _full_action(workload, trusted):
    return {**workload, **trusted}


@pytest.fixture
def breaker():
    return NumerailSystemLocal(_make_config())


# ═══════════════════════════════════════════════════════════════════════
#  A. TRUSTED RESERVE OVERRIDE
# ═══════════════════════════════════════════════════════════════════════

class TestTrustedReserveOverride:
    """Agent-supplied reserve/margin values are overwritten by trusted context."""

    def test_reserve_fields_overwritten(self, breaker):
        trusted = _default_trusted()
        fake = {**_safe_workload(),
                "current_gpu_util": 0.05, "current_api_util": 0.05,
                "ctrl_gpu_reserve_seconds": 1, "ctrl_api_reserve_calls": 0,
                "ctrl_parallel_reserve": 0,
                "gpu_disturbance_margin_seconds": 0, "api_disturbance_margin_calls": 0}

        result = breaker.enforce(fake, action_id="ta1", trusted_context=trusted)
        merged = result["feedback"].get("merged_values", {})
        assert merged["ctrl_gpu_reserve_seconds"] == 30
        assert merged["ctrl_api_reserve_calls"] == 4
        assert merged["ctrl_parallel_reserve"] == 2
        assert merged["gpu_disturbance_margin_seconds"] == 15
        assert merged["api_disturbance_margin_calls"] == 3
        assert merged["current_gpu_util"] == 0.30
        assert merged["current_api_util"] == 0.20


# ═══════════════════════════════════════════════════════════════════════
#  B. RESERVE-PRESERVING APPROVE
# ═══════════════════════════════════════════════════════════════════════

class TestReservePreservingApprove:
    """A safe workload is admitted and satisfies all reserve-aware constraints."""

    def test_safe_grant_approved(self, breaker):
        trusted = _default_trusted()
        result = breaker.enforce(
            _full_action(_safe_workload(), trusted),
            action_id="tb1", trusted_context=trusted)
        assert result["decision"] == "approve"

    def test_approved_grant_preserves_reserve(self, breaker):
        trusted = _default_trusted()
        result = breaker.enforce(
            _full_action(_safe_workload(), trusted),
            action_id="tb2", trusted_context=trusted)
        ev = result["enforced_values"]
        gpu_h = 0.30 + ev["gpu_seconds"]/240 + 30/240 + 15/240
        api_h = 0.20 + ev["external_api_calls"]/40 + 4/40 + 3/40
        par_h = ev["parallel_workers"] + 2
        assert gpu_h <= 0.90 + 1e-6
        assert api_h <= 0.90 + 1e-6
        assert par_h <= 16 + 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  C. RESERVE-STARVATION REJECT
# ═══════════════════════════════════════════════════════════════════════

class TestReserveStarvationReject:
    """Workload that would consume the controller's reserve is blocked."""

    def test_max_gpu_rejected_with_reserves(self, breaker):
        """gpu_seconds=120 passes old policy (0.80 ≤ 0.90) but fails with reserves."""
        trusted = _default_trusted()
        workload = {**_safe_workload(), "gpu_seconds": 120}
        result = breaker.enforce(
            _full_action(workload, trusted),
            action_id="tc1", trusted_context=trusted)
        assert result["decision"] == "reject"

    def test_old_headroom_would_pass(self):
        """Confirm the same request satisfies the old-style headroom constraint."""
        old_headroom = 0.30 + 120 / 240
        assert old_headroom <= 0.90

    def test_new_headroom_fails(self):
        """Confirm reserve-aware headroom rejects it."""
        new_headroom = 0.30 + 120/240 + 30/240 + 15/240
        assert new_headroom > 0.90


# ═══════════════════════════════════════════════════════════════════════
#  D. STRUCTURAL PROJECTION WITH RESERVE INTACT
# ═══════════════════════════════════════════════════════════════════════

class TestStructuralProjectionReserveIntact:
    """external_api > tool_calls projects while trusted fields stay unchanged."""

    def test_projects_structural_violation(self, breaker):
        trusted = _light_trusted()
        workload = {"prompt_k": 16, "completion_k": 4, "tool_calls": 4,
                     "external_api_calls": 8, "gpu_seconds": 30, "parallel_workers": 2}
        result = breaker.enforce(
            _full_action(workload, trusted),
            action_id="td1", trusted_context=trusted)
        assert result["decision"] == "project"
        ev = result["enforced_values"]
        assert ev["external_api_calls"] <= ev["tool_calls"] + 1e-6

    def test_reserve_fields_unchanged_after_projection(self, breaker):
        trusted = _light_trusted()
        workload = {"prompt_k": 16, "completion_k": 4, "tool_calls": 4,
                     "external_api_calls": 8, "gpu_seconds": 30, "parallel_workers": 2}
        result = breaker.enforce(
            _full_action(workload, trusted),
            action_id="td2", trusted_context=trusted)
        ev = result["enforced_values"]
        assert abs(ev["ctrl_gpu_reserve_seconds"] - 15) < 0.01
        assert abs(ev["ctrl_api_reserve_calls"] - 2) < 0.01
        assert abs(ev["ctrl_parallel_reserve"] - 1) < 0.01
        assert abs(ev["gpu_disturbance_margin_seconds"] - 8) < 0.01
        assert abs(ev["api_disturbance_margin_calls"] - 2) < 0.01


# ═══════════════════════════════════════════════════════════════════════
#  E. BOUNDARY TEST
# ═══════════════════════════════════════════════════════════════════════

class TestBoundary:
    """Request at the reserve-aware admissible boundary."""

    def test_boundary_gpu_headroom_binds(self, breaker):
        """gpu_seconds=99 makes gpu_headroom = 0.30 + 99/240 + 30/240 + 15/240 = 0.90 exactly."""
        trusted = _default_trusted()
        workload = {"prompt_k": 16, "completion_k": 4, "tool_calls": 5,
                     "external_api_calls": 2, "gpu_seconds": 99.0, "parallel_workers": 4}
        result = breaker.enforce(
            _full_action(workload, trusted),
            action_id="te1", trusted_context=trusted)
        assert result["decision"] in ("approve", "project")
        ev = result["enforced_values"]
        gpu_h = 0.30 + ev["gpu_seconds"]/240 + 30/240 + 15/240
        assert gpu_h <= 0.90 + 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  F. WORST-CASE RESERVE CHECK
# ═══════════════════════════════════════════════════════════════════════

class TestWorstCaseReserve:
    """Explicitly compute residual reserve and assert the controller is protected."""

    def test_gpu_reserve_preserved_after_enforcement(self, breaker):
        trusted = _default_trusted()
        result = breaker.enforce(
            _full_action(_safe_workload(), trusted),
            action_id="tf1", trusted_context=trusted)
        ev = result["enforced_values"]
        gpu_h = (trusted["current_gpu_util"]
                 + ev["gpu_seconds"] / 240
                 + trusted["ctrl_gpu_reserve_seconds"] / 240
                 + trusted["gpu_disturbance_margin_seconds"] / 240)
        assert gpu_h <= 0.90 + 1e-6

    def test_api_reserve_preserved_after_enforcement(self, breaker):
        trusted = _default_trusted()
        result = breaker.enforce(
            _full_action(_safe_workload(), trusted),
            action_id="tf2", trusted_context=trusted)
        ev = result["enforced_values"]
        api_h = (trusted["current_api_util"]
                 + ev["external_api_calls"] / 40
                 + trusted["ctrl_api_reserve_calls"] / 40
                 + trusted["api_disturbance_margin_calls"] / 40)
        assert api_h <= 0.90 + 1e-6

    def test_parallel_reserve_preserved_after_enforcement(self, breaker):
        trusted = _default_trusted()
        result = breaker.enforce(
            _full_action(_safe_workload(), trusted),
            action_id="tf3", trusted_context=trusted)
        ev = result["enforced_values"]
        par_h = ev["parallel_workers"] + trusted["ctrl_parallel_reserve"]
        assert par_h <= 16.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  G. BUDGET ISOLATION
# ═══════════════════════════════════════════════════════════════════════

class TestBudgetIsolation:
    """Budgets decrement only on workload dimensions, not reserve/margin."""

    def test_gpu_budget_tracks_gpu_seconds_only(self, breaker):
        trusted = _default_trusted()
        breaker.enforce(
            _full_action({**_safe_workload(), "gpu_seconds": 50}, trusted),
            action_id="tg1", trusted_context=trusted)
        remaining = breaker.budget_remaining
        assert remaining["gpu_day"] == pytest.approx(550.0)

    def test_api_budget_tracks_external_api_only(self, breaker):
        trusted = _default_trusted()
        breaker.enforce(
            _full_action({**_safe_workload(), "external_api_calls": 7}, trusted),
            action_id="tg2", trusted_context=trusted)
        remaining = breaker.budget_remaining
        assert remaining["api_day"] == pytest.approx(73.0)

    def test_reserve_fields_do_not_affect_budgets(self, breaker):
        """Two enforcements with different trusted reserves produce same budget consumption."""
        trusted_a = {**_default_trusted(), "ctrl_gpu_reserve_seconds": 10,
                     "gpu_disturbance_margin_seconds": 5}
        trusted_b = {**_default_trusted(), "ctrl_gpu_reserve_seconds": 30,
                     "gpu_disturbance_margin_seconds": 15}
        workload = {**_safe_workload(), "gpu_seconds": 50, "external_api_calls": 3}

        breaker.enforce(_full_action(workload, trusted_a),
                        action_id="tg3a", trusted_context=trusted_a)
        after_a = dict(breaker.budget_remaining)

        breaker.enforce(_full_action(workload, trusted_b),
                        action_id="tg3b", trusted_context=trusted_b)
        after_b = dict(breaker.budget_remaining)

        # Second enforcement should consume the same delta as the first
        gpu_delta_a = 600.0 - after_a["gpu_day"]
        gpu_delta_b = after_a["gpu_day"] - after_b["gpu_day"]
        assert gpu_delta_a == pytest.approx(gpu_delta_b)
