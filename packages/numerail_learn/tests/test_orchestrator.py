"""Tests for EnforcementRLOrchestrator."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from numerail_learn.experience import EnforcementExperienceBuffer
from numerail_learn.orchestrator import (
    EnforcementRLOrchestrator,
    EpisodeStats,
)
from numerail_learn.reward import conservative_shaper, EnforcementRewardShaper


# ---------------------------------------------------------------------------
# Mock GovernedStep
# ---------------------------------------------------------------------------


class _MockBreakerMode:
    def __init__(self, val: str):
        self.value = val


@dataclass
class _MockBreaker:
    mode: _MockBreakerMode
    overload_score: float = 0.25
    reason: str = "test"


@dataclass
class _MockGovernedStep:
    numerail_result: Dict[str, Any]
    breaker: _MockBreaker
    grant: Optional[Any] = None
    envelope: Optional[Any] = None


def _make_step(decision: str = "approve", breaker_mode: str = "closed",
               action_id: str = "act-001") -> _MockGovernedStep:
    enforced = {"gpu_seconds": 8.0, "api_calls": 4.0} if decision in ("approve", "project") else None
    dist = 2.0 if decision == "project" else 0.0
    return _MockGovernedStep(
        numerail_result={
            "decision":       decision,
            "enforced_values": enforced,
            "action_id":      action_id,
            "feedback":       [("gpu_limit", 0.5)] if decision == "project" else [],
            "solver_method":  "cvxpy",
            "distance":       dist,
        },
        breaker=_MockBreaker(mode=_MockBreakerMode(breaker_mode), overload_score=0.3),
    )


# ---------------------------------------------------------------------------
# Mock governor / backend
# ---------------------------------------------------------------------------


def _make_governor(budget: Optional[Dict[str, float]] = None):
    gov = MagicMock()
    gov.backend.budget_remaining.return_value = budget or {
        "gpu_shift": 3600.0,
        "external_api_shift": 500.0,
        "mutation_shift": 100.0,
    }
    return gov


def _orch(shaper: Optional[EnforcementRewardShaper] = None,
          budget: Optional[Dict[str, float]] = None) -> EnforcementRLOrchestrator:
    buf = EnforcementExperienceBuffer(max_size=500)
    gov = _make_governor(budget)
    s   = shaper or conservative_shaper()
    return EnforcementRLOrchestrator(
        governor=gov,
        buffer=buf,
        reward_shaper=s,
        schema_fields=["gpu_seconds", "api_calls"],
        budget_initial={"gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0},
    )


def _ctx():
    return [{"role": "user", "content": "run task"}]


def _tc():
    return {"name": "execute", "arguments": {"gpu_seconds": 10.0, "api_calls": 5.0}}


def _pv():
    return np.array([10.0, 5.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_step_from_governed_step():
    orch = _orch()
    step = _make_step("approve")
    exp  = orch.record_step(_ctx(), _tc(), step, _pv())
    assert exp.result == "approve"
    assert len(orch._buffer) == 1


def test_record_episode_boundary():
    orch = _orch()
    for _ in range(7):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("reject"), _pv())
    stats = orch.record_episode_boundary()
    assert isinstance(stats, EpisodeStats)
    assert stats.total_actions == 10
    assert stats.approve_count == 7
    assert stats.reject_count  == 3
    assert abs(stats.approval_rate - 0.7) < 1e-9


def test_approval_rate_history():
    orch = _orch()
    # Episode 1: all approve
    for _ in range(5):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    orch.record_episode_boundary()
    # Episode 2: half approve
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("reject"), _pv())
    orch.record_episode_boundary()

    history = orch.approval_rate_history
    assert len(history) == 2
    # First episode: rate=1.0, second episode lower
    assert abs(history[0][1] - 1.0) < 1e-9
    assert history[1][1] < 1.0


def test_export_sft_data():
    orch = _orch()
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("project"), _pv())
    sft = orch.export_sft_data()
    assert len(sft) == 3  # only projects


def test_export_dpo_data():
    orch = _orch()
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("approve", breaker_mode="closed"), _pv())
    for _ in range(3):
        orch.record_step(_ctx(), _tc(), _make_step("reject",  breaker_mode="closed"), _pv())
    dpo = orch.export_dpo_data()
    assert len(dpo) == 3


def test_export_ppo_data():
    orch = _orch()
    for d in ["approve", "project", "reject", "approve", "approve"]:
        step = _make_step(d)
        orch.record_step(_ctx(), _tc(), step, _pv())
    ppo = orch.export_ppo_data()
    assert len(ppo) == 5


def test_export_analytics():
    orch = _orch()
    for _ in range(5):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    analytics = orch.export_analytics()
    assert "result" in analytics
    assert len(analytics["result"]) == 5


def test_dimension_report():
    orch = _orch()
    # Record some project steps that produce violations
    for _ in range(4):
        orch.record_step(_ctx(), _tc(), _make_step("project"), _pv())
    report = orch.dimension_report()
    assert "violation_frequency" in report
    assert "most_violated" in report
    assert "recommendation" in report
    assert isinstance(report["violation_frequency"], dict)


def test_improvement_report():
    orch = _orch()
    # Episode 1: low approval
    for _ in range(2):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(8):
        orch.record_step(_ctx(), _tc(), _make_step("reject"), _pv())
    orch.record_episode_boundary()

    # Episode 2: high approval
    for _ in range(9):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(1):
        orch.record_step(_ctx(), _tc(), _make_step("reject"), _pv())
    orch.record_episode_boundary()

    report = orch.improvement_report()
    assert report["episodes_completed"] == 2
    assert report["initial_approval_rate"] < report["current_approval_rate"]
    assert len(report["approval_rate_trend"]) == 2


def test_save_load_state():
    orch = _orch()
    for _ in range(5):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    orch.record_episode_boundary()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state")
        orch.save_state(path)

        orch2 = _orch()
        orch2.load_state(path)

        assert len(orch2._buffer) == 5
        assert orch2._episode_count == 1
        assert len(orch2.approval_rate_history) == 1


# ---------------------------------------------------------------------------
# Retraction + boundary proximity tests
# ---------------------------------------------------------------------------


def test_export_sft_with_retraction():
    """export_sft_data applies retraction using the most recent APPROVE as reference."""
    buf = EnforcementExperienceBuffer(max_size=500)
    gov = _make_governor()
    orch = EnforcementRLOrchestrator(
        governor=gov, buffer=buf, reward_shaper=conservative_shaper(),
        schema_fields=["gpu_seconds", "api_calls"],
        budget_initial={"gpu_shift": 3600.0},
        retraction_factor=0.2,
    )

    # Record 2 approves then 2 projects
    for _ in range(2):
        orch.record_step(_ctx(), _tc(), _make_step("approve"), _pv())
    for _ in range(2):
        orch.record_step(_ctx(), _tc(), _make_step("project"), _pv())

    sft = orch.export_sft_data()
    assert len(sft) == 2
    # With an APPROVE in buffer, retraction should be applied
    for ex in sft:
        assert ex["retraction_applied"] is True
        assert ex["retraction_factor"] == pytest.approx(0.2)


def test_boundary_proximity_report():
    """boundary_proximity_report flags dimensions where proposals are near the cap."""
    buf = EnforcementExperienceBuffer(max_size=500)
    gov = _make_governor()
    orch = EnforcementRLOrchestrator(
        governor=gov, buffer=buf, reward_shaper=conservative_shaper(),
        schema_fields=["gpu_seconds", "api_calls"],
        budget_initial={"gpu_shift": 3600.0},
    )

    # Record APPROVE steps where proposed ≈ enforced (≈ at the cap)
    # gpu_seconds: proposed=9.5, enforced=10.0 → fraction 95% → boundary-seeking
    # api_calls:   proposed=2.0, enforced=10.0 → fraction 20% → not boundary-seeking
    import dataclasses
    for _ in range(5):
        step = _make_step("approve")
        # Override numerail_result enforced values to simulate caps
        step.numerail_result["enforced_values"] = {"gpu_seconds": 10.0, "api_calls": 10.0}
        pv = np.array([9.5, 2.0], dtype=np.float64)
        ev = np.array([10.0, 10.0], dtype=np.float64)
        orch.record_step(_ctx(), _tc(), step, pv)
        # Patch enforced_vector in the buffer so boundary_proximity_report can read it
        with buf._lock:
            for i in range(len(buf._buffer)):
                if buf._buffer[i].result == "approve":
                    buf._buffer[i] = dataclasses.replace(buf._buffer[i], enforced_vector=ev)

    report = orch.boundary_proximity_report()
    assert "mean_cap_fraction" in report
    assert "boundary_seeking_dimensions" in report
    assert "recommendation" in report
    # gpu_seconds at 95% → boundary-seeking
    assert "gpu_seconds" in report["boundary_seeking_dimensions"]
    # api_calls at 20% → not boundary-seeking
    assert "api_calls" not in report["boundary_seeking_dimensions"]
