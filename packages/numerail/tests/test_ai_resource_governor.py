"""Tests for the AI resource governor example.

Tests the production-shaped path through NumerailSystemLocal: policy parse,
scope checks, trusted-context injection, budget handling, ledger, audit,
outbox, and rollback.
"""

import numpy as np
import pytest

from numerail.engine import RollbackResult
from numerail.errors import AuthorizationError
from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_config():
    """Build the AI resource governor policy config."""
    fields = [
        "prompt_k", "completion_k", "tool_calls", "external_api_calls",
        "gpu_seconds", "parallel_workers", "current_gpu_util", "current_api_util",
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
    }

    A_rows, b_rows, names = [], [], []

    for f in fields:
        A_rows.append(row({f: 1.0}))
        b_rows.append(float(maxes[f]))
        names.append(f"max_{f}")

    for f in fields:
        A_rows.append(row({f: -1.0}))
        b_rows.append(0.0)
        names.append(f"min_{f}")

    A_rows.append(row({"external_api_calls": 1, "tool_calls": -1}))
    b_rows.append(0.0)
    names.append("external_le_tool_calls")

    A_rows.append(row({"current_gpu_util": 1, "gpu_seconds": 1 / 240}))
    b_rows.append(0.90)
    names.append("gpu_headroom_linear")

    A_rows.append(row({"current_api_util": 1, "external_api_calls": 1 / 40}))
    b_rows.append(0.90)
    names.append("api_headroom_linear")

    A_rows.append(row({"gpu_seconds": 1}))
    b_rows.append(600.0)
    names.append("remaining_gpu_day")

    A_rows.append(row({"external_api_calls": 1}))
    b_rows.append(80.0)
    names.append("remaining_api_day")

    Q_diag = [1/64**2, 1/16**2, 1/40**2, 1/20**2, 1/120**2, 1/16**2, 0, 0]

    M_socp = np.zeros((3, n))
    M_socp[0, idx["current_gpu_util"]] = 1.0
    M_socp[0, idx["gpu_seconds"]] = 1 / 240
    M_socp[1, idx["current_api_util"]] = 1.0
    M_socp[1, idx["external_api_calls"]] = 1 / 40
    M_socp[2, idx["parallel_workers"]] = 1 / 16

    A0_psd = np.eye(2).tolist()
    A_psd_list = [
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, -1/40]],
        [[-1/240, 0], [0, 0]],
        [[0, 0.35/16], [0.35/16, 0]],
        [[-1, 0], [0, 0]],
        [[0, 0], [0, -1]],
    ]

    return {
        "policy_id": "ai_resource_governor_v1",
        "schema": {"fields": fields},
        "polytope": {
            "A": [list(r) for r in A_rows],
            "b": b_rows,
            "names": names,
        },
        "quadratic_constraints": [{
            "Q": np.diag(Q_diag).tolist(),
            "a": [0] * n, "b": 2.25,
            "name": "resource_energy",
        }],
        "socp_constraints": [{
            "M": M_socp.tolist(), "q": [0] * 3, "c": [0] * n, "d": 1.15,
            "name": "burst_envelope",
        }],
        "psd_constraints": [{
            "A0": A0_psd, "A_list": A_psd_list,
            "name": "coupled_headroom_psd",
        }],
        "trusted_fields": ["current_gpu_util", "current_api_util"],
        "enforcement": {
            "mode": "project",
            "dimension_policies": {
                "current_gpu_util": "forbidden",
                "current_api_util": "forbidden",
                "parallel_workers": "project_with_flag",
            },
            "routing_thresholds": {
                "silent": 0.05, "flagged": 1.5,
                "confirmation": 4.0, "hard_reject": 8.0,
            },
        },
        "budgets": [
            {"name": "gpu_day", "constraint_name": "remaining_gpu_day",
             "weight": {"gpu_seconds": 1.0}, "initial": 600.0,
             "consumption_mode": "nonnegative"},
            {"name": "api_day", "constraint_name": "remaining_api_day",
             "weight": {"external_api_calls": 1.0}, "initial": 80.0,
             "consumption_mode": "nonnegative"},
        ],
    }


def _safe_grant():
    return {
        "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
        "external_api_calls": 5, "gpu_seconds": 60, "parallel_workers": 4,
        "current_gpu_util": 0.30, "current_api_util": 0.20,
    }


def _safe_trusted():
    return {"current_gpu_util": 0.30, "current_api_util": 0.20}


@pytest.fixture
def governor():
    return NumerailSystemLocal(_make_config())


# ═══════════════════════════════════════════════════════════════════════
#  TRUSTED TELEMETRY
# ═══════════════════════════════════════════════════════════════════════

class TestTrustedTelemetry:
    """Trusted context overwrites agent-supplied telemetry."""

    def test_trusted_overrides_agent_claims(self, governor):
        result = governor.enforce(
            {**_safe_grant(), "current_gpu_util": 0.10, "current_api_util": 0.10},
            action_id="t1",
            trusted_context={"current_gpu_util": 0.30, "current_api_util": 0.20},
        )
        merged = result["feedback"].get("merged_values", {})
        assert merged["current_gpu_util"] == 0.30
        assert merged["current_api_util"] == 0.20

    def test_spoof_low_util_causes_reject(self, governor):
        """Agent claims low utilization; real utilization makes the request infeasible."""
        result = governor.enforce(
            {**_safe_grant(), "gpu_seconds": 80,
             "current_gpu_util": 0.10, "current_api_util": 0.10},
            action_id="t2",
            trusted_context={"current_gpu_util": 0.65, "current_api_util": 0.55},
        )
        assert result["decision"] == "reject"

    def test_ledger_preserves_raw_and_merged(self, governor):
        governor.enforce(
            {**_safe_grant(), "current_gpu_util": 0.05},
            action_id="t3",
            trusted_context={"current_gpu_util": 0.30, "current_api_util": 0.20},
        )
        ledger = governor.ledger
        entry = ledger["t3"]
        assert entry["raw_values"]["current_gpu_util"] == 0.05
        assert entry["merged_values"]["current_gpu_util"] == 0.30


# ═══════════════════════════════════════════════════════════════════════
#  APPROVE / PROJECT / REJECT
# ═══════════════════════════════════════════════════════════════════════

class TestEnforcementDecisions:

    def test_safe_grant_approved(self, governor):
        result = governor.enforce(
            _safe_grant(), action_id="d1",
            trusted_context=_safe_trusted(),
        )
        assert result["decision"] == "approve"

    def test_overlarge_gpu_rejected_under_real_util(self, governor):
        """High live utilization makes a large GPU request infeasible."""
        result = governor.enforce(
            {**_safe_grant(), "gpu_seconds": 120, "parallel_workers": 8,
             "current_gpu_util": 0.50, "current_api_util": 0.50},
            action_id="d2",
            trusted_context={"current_gpu_util": 0.80, "current_api_util": 0.75},
        )
        assert result["decision"] == "reject"

    def test_external_exceeds_tool_calls_projects(self, governor):
        """external_api_calls > tool_calls is projected to equality."""
        result = governor.enforce(
            {**_safe_grant(), "tool_calls": 4, "external_api_calls": 8},
            action_id="d3",
            trusted_context=_safe_trusted(),
        )
        assert result["decision"] == "project"
        ev = result["enforced_values"]
        assert ev["external_api_calls"] <= ev["tool_calls"] + 1e-6

    def test_project_result_satisfies_structural_constraint(self, governor):
        """After projection, external_api_calls ≤ tool_calls holds."""
        result = governor.enforce(
            {**_safe_grant(), "tool_calls": 4, "external_api_calls": 8},
            action_id="d4",
            trusted_context=_safe_trusted(),
        )
        ev = result["enforced_values"]
        assert ev["external_api_calls"] <= ev["tool_calls"] + 1e-6


# ═══════════════════════════════════════════════════════════════════════
#  BUDGETS
# ═══════════════════════════════════════════════════════════════════════

class TestBudgets:

    def test_budgets_deplete_across_grants(self, governor):
        governor.enforce(
            _safe_grant(), action_id="b1",
            trusted_context=_safe_trusted(),
        )
        governor.enforce(
            {**_safe_grant(), "gpu_seconds": 40, "external_api_calls": 3},
            action_id="b2",
            trusted_context=_safe_trusted(),
        )
        remaining = governor.budget_remaining
        assert remaining["gpu_day"] == pytest.approx(500.0)
        assert remaining["api_day"] == pytest.approx(72.0)

    def test_rollback_restores_budgets(self, governor):
        governor.enforce(
            _safe_grant(), action_id="b3",
            trusted_context=_safe_trusted(),
        )
        pre = dict(governor.budget_remaining)
        governor.enforce(
            {**_safe_grant(), "gpu_seconds": 40, "external_api_calls": 3},
            action_id="b4",
            trusted_context=_safe_trusted(),
        )
        governor.rollback("b4")
        post = governor.budget_remaining
        assert post["gpu_day"] == pytest.approx(pre["gpu_day"])
        assert post["api_day"] == pytest.approx(pre["api_day"])

    def test_rollback_returns_result_type(self, governor):
        governor.enforce(
            _safe_grant(), action_id="b5",
            trusted_context=_safe_trusted(),
        )
        rb = governor.rollback("b5")
        assert isinstance(rb, RollbackResult)
        assert rb.rolled_back is True
        assert rb.audit_hash is not None


# ═══════════════════════════════════════════════════════════════════════
#  OUTBOX
# ═══════════════════════════════════════════════════════════════════════

class TestOutbox:

    def test_outbox_on_approve(self, governor):
        governor.enforce(
            _safe_grant(), action_id="o1",
            trusted_context=_safe_trusted(),
            execution_topic="runtime",
        )
        assert len(governor.outbox_events) == 1
        assert governor.outbox_events[0]["topic"] == "runtime"
        assert governor.outbox_events[0]["action_id"] == "o1"

    def test_outbox_on_project(self, governor):
        governor.enforce(
            {**_safe_grant(), "tool_calls": 4, "external_api_calls": 8},
            action_id="o2",
            trusted_context=_safe_trusted(),
            execution_topic="runtime",
        )
        assert len(governor.outbox_events) == 1

    def test_no_outbox_on_reject(self, governor):
        governor.enforce(
            {**_safe_grant(), "gpu_seconds": 120, "parallel_workers": 8},
            action_id="o3",
            trusted_context={"current_gpu_util": 0.80, "current_api_util": 0.75},
            execution_topic="runtime",
        )
        assert len(governor.outbox_events) == 0

    def test_no_outbox_without_topic(self, governor):
        governor.enforce(
            _safe_grant(), action_id="o4",
            trusted_context=_safe_trusted(),
        )
        assert len(governor.outbox_events) == 0


# ═══════════════════════════════════════════════════════════════════════
#  AUDIT
# ═══════════════════════════════════════════════════════════════════════

class TestAudit:

    def test_audit_records_created(self, governor):
        governor.enforce(
            _safe_grant(), action_id="a1",
            trusted_context=_safe_trusted(),
        )
        assert len(governor.audit_records) == 1
        assert governor.audit_records[0]["type"] == "decision"

    def test_rollback_adds_audit_record(self, governor):
        governor.enforce(
            _safe_grant(), action_id="a2",
            trusted_context=_safe_trusted(),
        )
        governor.rollback("a2")
        assert len(governor.audit_records) == 2
        assert governor.audit_records[1]["type"] == "rollback"

    def test_audit_hashes_are_linked(self, governor):
        governor.enforce(
            _safe_grant(), action_id="a3",
            trusted_context=_safe_trusted(),
        )
        governor.enforce(
            {**_safe_grant(), "gpu_seconds": 40},
            action_id="a4",
            trusted_context=_safe_trusted(),
        )
        r1, r2 = governor.audit_records
        assert r2["prev_hash"] == r1["hash"]
