"""Tests for EnforcementRewardShaper and preset configurations."""

from __future__ import annotations

import numpy as np
import pytest

from numerail_learn.experience import EnforcementExperienceBuffer, EnforcementExperience
from numerail_learn.reward import (
    EnforcementRewardShaper,
    conservative_shaper,
    permissive_shaper,
    strict_shaper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shaper(**kwargs) -> EnforcementRewardShaper:
    defaults = dict(
        approve_reward=1.0, project_base=0.0, reject_penalty=-1.0,
        distance_scale=0.1, violation_scale=0.5, budget_bonus_scale=0.05,
    )
    defaults.update(kwargs)
    return EnforcementRewardShaper(**defaults)


def _exp(result: str, distance: float = 0.0, violations=None,
         breaker: str = "closed") -> EnforcementExperience:
    pv = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    ev = np.array([8.0, 4.0, 2.0], dtype=np.float64) if result == "project" else None
    return EnforcementExperience(
        experience_id="exp-000001",
        action_id="act-001",
        timestamp_ms=0.0,
        conversation_context=[],
        tool_call={"name": "act", "arguments": {"a": 10.0, "b": 5.0, "c": 2.0}},
        proposed_vector=pv,
        result=result,
        enforced_vector=ev,
        distance=distance,
        violations=violations or [],
        solver_method="",
        routing_decision=None,
        breaker_mode=breaker,
        budget_remaining={"gpu_shift": 1000.0},
        overload_score=0.3,
        policy_digest="abc",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_approve_reward():
    s = _shaper()
    r = s.compute_reward("approve", 0.0, [])
    assert abs(r - 1.0) < 1e-9


def test_reject_penalty():
    s = _shaper()
    r = s.compute_reward("reject", -1.0, [])
    assert r == pytest.approx(-1.0)


def test_project_distance_penalty():
    s = _shaper(project_base=0.0, distance_scale=0.1)
    r_near = s.compute_reward("project", 1.0, [])
    r_far  = s.compute_reward("project", 5.0, [])
    assert r_near > r_far


def test_violation_penalty():
    s = _shaper(violation_scale=0.5)
    r_none = s.compute_reward("project", 0.0, [])
    r_some = s.compute_reward("project", 0.0, [("lim", 1.0), ("lim2", 1.0)])
    assert r_none > r_some


def test_budget_bonus():
    s = _shaper(budget_bonus_scale=0.1)
    r_full = s.compute_reward("approve", 0.0, [],
                               budget_remaining={"gpu": 1000.0},
                               budget_initial={"gpu": 1000.0})
    r_none = s.compute_reward("approve", 0.0, [],
                               budget_remaining={"gpu": 0.0},
                               budget_initial={"gpu": 1000.0})
    assert r_full > r_none


def test_detailed_reward_components():
    s = _shaper()
    d = s.compute_detailed_reward("project", 2.0, [("lim", 1.0)])
    total = (d["approval_component"] + d["distance_component"] +
             d["violation_component"] + d["budget_component"])
    assert abs(total - d["total_reward"]) < 1e-9


def test_dimension_feedback_on_project():
    s = _shaper()
    pv = np.array([10.0, 5.0], dtype=np.float64)
    ev = np.array([8.0,  4.0], dtype=np.float64)
    d = s.compute_detailed_reward(
        "project", 2.449, [],
        proposed_vector=pv, enforced_vector=ev,
        schema_fields=["gpu_seconds", "api_calls"],
    )
    fb = d["dimension_feedback"]
    assert "gpu_seconds" in fb
    assert abs(fb["gpu_seconds"] - 2.0) < 1e-6
    assert "api_calls" in fb
    assert abs(fb["api_calls"] - 1.0) < 1e-6


def test_dimension_feedback_not_on_approve():
    s = _shaper()
    pv = np.array([10.0, 5.0], dtype=np.float64)
    ev = np.array([10.0, 5.0], dtype=np.float64)  # same → APPROVE
    d = s.compute_detailed_reward(
        "approve", 0.0, [],
        proposed_vector=pv, enforced_vector=ev,
        schema_fields=["gpu_seconds", "api_calls"],
    )
    # APPROVE → result != 'project' so dimension_feedback should be empty
    assert d["dimension_feedback"] == {}


def test_shape_experience():
    s = _shaper()
    exp = _exp("project", distance=2.0, violations=[("lim", 0.5)])
    original_reward = exp.reward
    shaped = s.shape_experience(exp, schema_fields=["a", "b", "c"])
    # Original unchanged
    assert exp.reward == original_reward
    # Shaped has populated reward
    assert shaped.reward != 0.0
    assert shaped.reward_components is not None


def test_conservative_shaper():
    s = conservative_shaper()
    assert s.reject_penalty <= -2.0
    # Project should be penalised
    r_project = s.compute_reward("project", 1.0, [])
    r_approve = s.compute_reward("approve", 0.0, [])
    assert r_approve > r_project


def test_permissive_shaper():
    s = permissive_shaper()
    assert s.reject_penalty > -1.0  # less harsh than default
    # Project base is positive
    assert s.project_base > 0.0


def test_strict_shaper():
    s = strict_shaper()
    assert s.reject_penalty <= -3.0
    # Project is heavily penalised
    r_project = s.compute_reward("project", 0.0, [])
    assert r_project <= -1.0
