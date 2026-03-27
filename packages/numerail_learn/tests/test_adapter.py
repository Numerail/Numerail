"""Tests for training data adapters."""

from __future__ import annotations

import numpy as np
import pytest

from numerail_learn.adapter import (
    corrected_tool_call,
    to_sft_examples,
    to_dpo_pairs,
    to_ppo_episodes,
    to_analytics_dataframe,
)
from numerail_learn.experience import EnforcementExperience


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
