"""Service tests: NumerailSystemLocal integration, scope checks, rollback."""

import pytest

from numerail.engine import EnforcementResult, RollbackResult
from numerail.errors import AuthorizationError
from numerail.protocols import ServiceRequest
from numerail.local import NumerailSystemLocal


@pytest.fixture
def local():
    return NumerailSystemLocal({
        "schema": {"fields": ["x", "y"]},
        "polytope": {
            "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
            "b": [1, 1, 0, 0],
            "names": ["ux", "uy", "lx", "ly"],
        },
        "enforcement": {"mode": "project"},
        "budgets": [{"name": "cap", "constraint_name": "ux",
                     "dimension_name": "x", "initial": 1.0}],
    })


@pytest.fixture
def local_no_budget():
    return NumerailSystemLocal({
        "schema": {"fields": ["x", "y"]},
        "polytope": {
            "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
            "b": [1, 1, 0, 0],
            "names": ["ux", "uy", "lx", "ly"],
        },
        "enforcement": {"mode": "project"},
    })


class TestLocalEnforce:
    def test_approve(self, local):
        r = local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        assert r["decision"] == "approve"

    def test_project(self, local):
        r = local.enforce({"x": 1.5, "y": 0.5}, action_id="t2")
        assert r["decision"] in ("approve", "project")

    def test_auto_action_id(self, local):
        r = local.enforce({"x": 0.5, "y": 0.5})
        assert r["action_id"].startswith("local_")

    def test_result_contains_all_keys(self, local):
        r = local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        assert "enforced_values" in r
        assert "decision" in r
        assert "feedback" in r
        assert "audit_hash" in r
        assert "action_id" in r


class TestLocalRollback:
    def test_rollback_returns_rollback_result(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        rb = local.rollback("t1")
        assert isinstance(rb, RollbackResult)
        assert rb.rolled_back is True
        assert rb.audit_hash is not None

    def test_rollback_bool(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        rb = local.rollback("t1")
        assert bool(rb) is True

    def test_double_rollback_raises(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        local.rollback("t1")
        with pytest.raises(ValueError, match="already rolled back"):
            local.rollback("t1")


class TestLocalBudget:
    def test_budget_remaining(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        remaining = local.budget_remaining
        assert "cap" in remaining
        assert remaining["cap"] < 1.0

    def test_rollback_restores_budget(self, local):
        before = local.budget_remaining["cap"]
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        local.rollback("t1")
        after = local.budget_remaining["cap"]
        assert after == pytest.approx(before)


class TestLocalAudit:
    def test_audit_records_populated(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        assert len(local.audit_records) == 1
        assert local.audit_records[0]["type"] == "decision"

    def test_rollback_adds_audit_record(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        local.rollback("t1")
        assert len(local.audit_records) == 2
        assert local.audit_records[1]["type"] == "rollback"


class TestLocalMetrics:
    def test_enforcement_metrics(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        m = local.metrics
        assert len(m["enforcements"]) == 1

    def test_rollback_metrics(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        local.rollback("t1")
        m = local.metrics
        assert len(m["rollbacks"]) == 1


class TestLocalOutbox:
    def test_outbox_on_approve(self, local):
        r = local.enforce({"x": 0.5, "y": 0.5}, action_id="t1",
                          execution_topic="orders")
        assert len(local.outbox_events) == 1
        assert local.outbox_events[0]["topic"] == "orders"


class TestScopeEnforcement:
    def test_trusted_context_without_scope_raises(self, local):
        with pytest.raises(AuthorizationError):
            local._service.enforce(
                policy_id="local",
                proposed_action={"x": 0.5, "y": 0.5},
                action_id="t99",
                request=ServiceRequest(
                    scopes=["enforce"],
                    trusted_context={"x": 0.9},
                ),
            )

    def test_trusted_context_with_scope_succeeds(self, local):
        r = local._service.enforce(
            policy_id="local",
            proposed_action={"x": 0.5, "y": 0.5},
            action_id="t99",
            request=ServiceRequest(
                scopes=["enforce", "trusted:inject"],
                trusted_context={"x": 0.9},
            ),
        )
        assert r["decision"] in ("approve", "project")

    def test_rollback_without_scope_raises(self, local):
        local.enforce({"x": 0.5, "y": 0.5}, action_id="t1")
        with pytest.raises(AuthorizationError):
            local._service.rollback(action_id="t1", scopes=["enforce"])


class TestWeightMapBudgetIntegration:
    def test_from_config_with_weight_map(self):
        local = NumerailSystemLocal({
            "schema": {"fields": ["a", "b"]},
            "polytope": {
                "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "b": [100, 100, 0, 0],
                "names": ["ma", "mb", "la", "lb"],
            },
            "enforcement": {"mode": "project"},
            "budgets": [{
                "name": "combo", "constraint_name": "ma",
                "weight": {"a": 0.6, "b": 0.4},
                "initial": 100.0,
            }],
        })
        r = local.enforce({"a": 10.0, "b": 20.0}, action_id="t1")
        assert r["decision"] in ("approve", "project")


# ── Regression tests for v4.1.0 bug fixes ───────────────────────────────


class TestBudgetDeltaRegression:
    """Regression for P0#1: service-layer budget delta was using cumulative
    consumed total instead of per-action delta, causing double-counting."""

    def test_sequential_enforcements_correct_remaining(self):
        local = NumerailSystemLocal({
            "schema": {"fields": ["x"]},
            "polytope": {"A": [[1], [-1]], "b": [1, 0], "names": ["max", "min"]},
            "enforcement": {"mode": "project"},
            "budgets": [{"name": "cap", "constraint_name": "max",
                         "dimension_name": "x", "initial": 1.0}],
        })
        local.enforce({"x": 0.3}, action_id="a1")
        assert local.budget_remaining["cap"] == pytest.approx(0.7)
        local.enforce({"x": 0.2}, action_id="a2")
        assert local.budget_remaining["cap"] == pytest.approx(0.5)

    def test_three_enforcements_then_rollback(self):
        local = NumerailSystemLocal({
            "schema": {"fields": ["x"]},
            "polytope": {"A": [[1], [-1]], "b": [1, 0], "names": ["max", "min"]},
            "enforcement": {"mode": "project"},
            "budgets": [{"name": "cap", "constraint_name": "max",
                         "dimension_name": "x", "initial": 1.0}],
        })
        local.enforce({"x": 0.2}, action_id="a1")
        local.enforce({"x": 0.3}, action_id="a2")
        local.enforce({"x": 0.1}, action_id="a3")
        assert local.budget_remaining["cap"] == pytest.approx(0.4)
        local.rollback("a2")
        assert local.budget_remaining["cap"] == pytest.approx(0.7)
