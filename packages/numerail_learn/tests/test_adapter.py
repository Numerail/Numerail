"""Tests for training data adapters."""

from __future__ import annotations

import numpy as np
import pytest

from numerail_learn.adapter import (
    corrected_tool_call,
    find_interior_reference,
    to_sft_examples,
    to_dpo_pairs,
    to_ppo_episodes,
    to_analytics_dataframe,
)
from numerail_learn.experience import EnforcementExperience, EnforcementExperienceBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exp(result: str, distance: float = 0.0, breaker: str = "closed",
         reward: float = 0.5, timestamp_ms: float = 1000.0,
         gpu_proposed: float = 10.0, gpu_enforced: float = 8.0) -> EnforcementExperience:
    pv = np.array([gpu_proposed, 5.0], dtype=np.float64)
    ev = None
    if result == "project":
        ev = np.array([gpu_enforced, 4.0], dtype=np.float64)
    elif result == "approve":
        ev = np.array([gpu_proposed, 5.0], dtype=np.float64)

    return EnforcementExperience(
        experience_id="exp-001",
        action_id="act-001",
        timestamp_ms=timestamp_ms,
        conversation_context=[{"role": "user", "content": "do something"}],
        tool_call={"name": "execute", "arguments": {"gpu_seconds": gpu_proposed, "api_calls": 5.0}},
        proposed_vector=pv,
        result=result,
        enforced_vector=ev,
        distance=distance,
        violations=[],
        solver_method="cvxpy",
        routing_decision=None,
        breaker_mode=breaker,
        budget_remaining={"gpu_shift": 1000.0, "external_api_shift": 200.0, "mutation_shift": 50.0},
        overload_score=0.3,
        policy_digest="abc",
        reward=reward,
        reward_components={"approval_component": reward},
    )


# ---------------------------------------------------------------------------
# corrected_tool_call
# ---------------------------------------------------------------------------


def test_corrected_tool_call():
    original = {"name": "execute", "arguments": {"gpu_seconds": 150.0, "api_calls": 10.0, "misc": "keep"}}
    enforced = {"gpu_seconds": 120.0}
    result = corrected_tool_call(original, enforced)
    assert result["arguments"]["gpu_seconds"] == 120.0
    assert result["arguments"]["api_calls"] == 10.0  # unchanged
    assert result["arguments"]["misc"] == "keep"     # unchanged


def test_corrected_tool_call_no_mutation():
    original = {"name": "execute", "arguments": {"gpu_seconds": 150.0}}
    orig_copy = {"name": "execute", "arguments": {"gpu_seconds": 150.0}}
    corrected_tool_call(original, {"gpu_seconds": 120.0})
    assert original == orig_copy  # original not mutated


# ---------------------------------------------------------------------------
# SFT adapter
# ---------------------------------------------------------------------------


def test_to_sft_examples_from_project():
    exps = [_exp("project", distance=2.0) for _ in range(3)]
    examples = to_sft_examples(exps)
    assert len(examples) == 3
    for ex in examples:
        assert "messages" in ex
        # Last message should be assistant with corrected tool call
        last_msg = ex["messages"][-1]
        assert last_msg["role"] == "assistant"
        assert "execute" in last_msg["content"] or "{" in last_msg["content"]
        assert "enforcement_distance" in ex


def test_to_sft_skips_approve_and_reject():
    exps = [
        _exp("approve"),
        _exp("project", distance=1.0),
        _exp("reject"),
        _exp("project", distance=2.0),
    ]
    examples = to_sft_examples(exps)
    assert len(examples) == 2  # only the two projects


# ---------------------------------------------------------------------------
# DPO adapter
# ---------------------------------------------------------------------------


def test_to_dpo_pairs():
    exps = (
        [_exp("approve", breaker="closed", timestamp_ms=1000.0 + i) for i in range(3)] +
        [_exp("reject",  breaker="closed", timestamp_ms=1010.0 + i) for i in range(3)]
    )
    pairs = to_dpo_pairs(exps)
    assert len(pairs) == 3
    for p in pairs:
        assert "chosen" in p
        assert "rejected" in p
        assert "prompt" in p
        # chosen_reward >= rejected_reward (approve > reject)
        assert p["chosen_reward"] >= p["rejected_reward"]


def test_to_dpo_pairs_different_modes_not_paired():
    exps = [
        _exp("approve", breaker="closed"),
        _exp("reject",  breaker="throttled"),
    ]
    pairs = to_dpo_pairs(exps)
    assert len(pairs) == 0  # different modes → no pair


# ---------------------------------------------------------------------------
# PPO adapter
# ---------------------------------------------------------------------------


def test_to_ppo_episodes():
    exps = [
        _exp("approve", reward=1.0),
        _exp("project", distance=1.0, reward=0.0),
        _exp("reject",  reward=-1.0),
        _exp("approve", reward=1.0),
        _exp("project", distance=2.0, reward=-0.2),
    ]
    episodes = to_ppo_episodes(exps)
    assert len(episodes) == 5
    for ep in episodes:
        assert "query" in ep
        assert "response" in ep
        assert "reward" in ep
        assert "reward_components" in ep


# ---------------------------------------------------------------------------
# Analytics adapter
# ---------------------------------------------------------------------------


def test_to_analytics_dataframe():
    exps = [_exp("approve") for _ in range(3)] + [_exp("project", distance=1.0) for _ in range(2)]
    df = to_analytics_dataframe(exps)
    expected_cols = {
        "experience_id", "timestamp_ms", "result", "distance", "reward",
        "breaker_mode", "overload_score", "n_violations",
        "top_violation_name", "top_violation_magnitude",
        "proposed_gpu_seconds", "enforced_gpu_seconds",
        "budget_gpu_remaining", "budget_api_remaining", "budget_mutation_remaining",
    }
    assert set(df.keys()) >= expected_cols
    # Each column should have 5 entries
    for col_vals in df.values():
        assert len(col_vals) == 5


# ---------------------------------------------------------------------------
# SFT retraction tests
# ---------------------------------------------------------------------------


def _project_exp_custom(gpu_proposed: float, gpu_enforced: float,
                        api_proposed: float, api_enforced: float) -> EnforcementExperience:
    """PROJECT experience with explicit proposed and enforced vectors."""
    pv = np.array([gpu_proposed, api_proposed], dtype=np.float64)
    ev = np.array([gpu_enforced, api_enforced], dtype=np.float64)
    return EnforcementExperience(
        experience_id="exp-ret",
        action_id="act-ret",
        timestamp_ms=2000.0,
        conversation_context=[{"role": "user", "content": "do work"}],
        tool_call={"name": "execute", "arguments": {"gpu_seconds": gpu_proposed, "api_calls": api_proposed}},
        proposed_vector=pv,
        result="project",
        enforced_vector=ev,
        distance=float(np.linalg.norm(pv - ev)),
        violations=[],
        solver_method="cvxpy",
        routing_decision=None,
        breaker_mode="closed",
        budget_remaining={"gpu_shift": 1000.0},
        overload_score=0.2,
        policy_digest="abc",
    )


def test_sft_retraction_applied():
    """retraction pulls target toward reference: enforced + f*(ref - enforced)."""
    exp = _project_exp_custom(gpu_proposed=15.0, gpu_enforced=10.0,
                               api_proposed=5.0,  api_enforced=0.0)
    # enforced = [10, 0], reference = [5, 5], factor = 0.2
    # retracted = [10, 0] + 0.2 * ([5, 5] - [10, 0]) = [10, 0] + [-1, 1] = [9, 1]
    reference = np.array([5.0, 5.0], dtype=np.float64)
    examples = to_sft_examples([exp], retraction_factor=0.2, reference_vector=reference,
                                schema_fields=["gpu_seconds", "api_calls"])
    assert len(examples) == 1
    ex = examples[0]
    assert ex["retraction_applied"] is True
    assert ex["retraction_factor"] == pytest.approx(0.2)
    # Check corrected tool call arguments
    import json
    corrected = json.loads(ex["messages"][-1]["content"])
    assert corrected["arguments"]["gpu_seconds"] == pytest.approx(9.0)
    assert corrected["arguments"]["api_calls"] == pytest.approx(1.0)


def test_sft_retraction_convexity():
    """Retracted target is between enforced and reference (convex combination)."""
    exp = _project_exp_custom(gpu_proposed=20.0, gpu_enforced=12.0,
                               api_proposed=10.0, api_enforced=6.0)
    reference = np.array([4.0, 2.0], dtype=np.float64)
    enforced  = np.array([12.0, 6.0], dtype=np.float64)
    examples = to_sft_examples([exp], retraction_factor=0.3, reference_vector=reference,
                                schema_fields=["gpu_seconds", "api_calls"])
    import json
    args = json.loads(examples[0]["messages"][-1]["content"])["arguments"]
    gpu_target = args["gpu_seconds"]
    api_target = args["api_calls"]
    # Target must lie between enforced and reference on each dimension
    lo_gpu, hi_gpu = min(enforced[0], reference[0]), max(enforced[0], reference[0])
    lo_api, hi_api = min(enforced[1], reference[1]), max(enforced[1], reference[1])
    assert lo_gpu - 1e-9 <= gpu_target <= hi_gpu + 1e-9
    assert lo_api - 1e-9 <= api_target <= hi_api + 1e-9


def test_sft_no_retraction_without_reference():
    """No reference_vector → use raw enforced_vector, retraction_applied=False."""
    exp = _project_exp_custom(gpu_proposed=15.0, gpu_enforced=10.0,
                               api_proposed=5.0,  api_enforced=4.0)
    examples = to_sft_examples([exp], retraction_factor=0.2, reference_vector=None,
                                schema_fields=["gpu_seconds", "api_calls"])
    ex = examples[0]
    assert ex["retraction_applied"] is False
    assert ex["retraction_factor"] == pytest.approx(0.0)
    import json
    args = json.loads(ex["messages"][-1]["content"])["arguments"]
    assert args["gpu_seconds"] == pytest.approx(10.0)
    assert args["api_calls"] == pytest.approx(4.0)


def test_sft_no_retraction_on_zero_factor():
    """retraction_factor=0.0 → target equals enforced_vector."""
    exp = _project_exp_custom(gpu_proposed=15.0, gpu_enforced=10.0,
                               api_proposed=5.0,  api_enforced=4.0)
    reference = np.array([3.0, 2.0], dtype=np.float64)
    examples = to_sft_examples([exp], retraction_factor=0.0, reference_vector=reference,
                                schema_fields=["gpu_seconds", "api_calls"])
    ex = examples[0]
    assert ex["retraction_applied"] is False
    import json
    args = json.loads(ex["messages"][-1]["content"])["arguments"]
    assert args["gpu_seconds"] == pytest.approx(10.0)
    assert args["api_calls"] == pytest.approx(4.0)


def test_find_interior_reference():
    """find_interior_reference returns the most recent APPROVE's proposed_vector."""
    exps = [
        _exp("approve", timestamp_ms=1000.0, gpu_proposed=5.0),
        _exp("project", timestamp_ms=2000.0),
        _exp("approve", timestamp_ms=3000.0, gpu_proposed=7.0),  # most recent
        _exp("project", timestamp_ms=4000.0),
        _exp("reject",  timestamp_ms=5000.0),
    ]
    ref = find_interior_reference(exps)
    assert ref is not None
    # Most recent APPROVE (timestamp 3000) has gpu_proposed=7.0 at index 0
    assert ref[0] == pytest.approx(7.0)


def test_find_interior_reference_empty():
    """Buffer with only reject/project → find_interior_reference returns None."""
    exps = [_exp("reject"), _exp("project", distance=1.0)]
    ref = find_interior_reference(exps)
    assert ref is None
