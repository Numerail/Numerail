"""Training data adapters.

Converts EnforcementExperience instances into training formats for LLMs:
- SFT  (Supervised Fine-Tuning)  — from PROJECT corrections
- DPO  (Direct Preference Opt.)  — from APPROVE vs REJECT pairs
- PPO  (Proximal Policy Opt.)    — shaped rewards for all experiences
- Analytics export               — columnar dict for pandas DataFrames
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from numerail_learn.experience import EnforcementExperience


# ---------------------------------------------------------------------------
# Tool call helpers
# ---------------------------------------------------------------------------


def corrected_tool_call(
    original_tool_call: Dict[str, Any],
    enforced_values: Dict[str, float],
) -> Dict[str, Any]:
    """Return a copy of *original_tool_call* with enforced values substituted.

    Only keys present in both ``original_tool_call["arguments"]`` and
    ``enforced_values`` are overwritten. The original is not mutated.
    """
    new_tc = {
        "name":      original_tool_call.get("name", ""),
        "arguments": dict(original_tool_call.get("arguments", {})),
    }
    for key, val in enforced_values.items():
        if key in new_tc["arguments"]:
            new_tc["arguments"][key] = val
    return new_tc


def _tool_call_to_text(tc: Dict[str, Any]) -> str:
    """Render a tool call dict to a canonical text representation."""
    return json.dumps(tc, sort_keys=True)


def _context_to_string(messages: List[Dict[str, Any]]) -> str:
    """Flatten a message list to a single string for PPO queries."""
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SFT adapter
# ---------------------------------------------------------------------------


def to_sft_examples(
    experiences: List[EnforcementExperience],
) -> List[Dict[str, Any]]:
    """Convert PROJECT experiences into supervised fine-tuning examples.

    Each example contains:

    - ``messages``: conversation context with a corrected assistant turn appended.
    - ``original_tool_call``: what the model actually proposed.
    - ``enforcement_distance``: how far off the model was.
    - ``corrected_fields``: field names that were changed.

    APPROVE experiences are already correct (no correction target needed).
    REJECT experiences have no correction target. Both are skipped.
    """
    examples: List[Dict[str, Any]] = []

    for exp in experiences:
        if exp.result != "project" or exp.enforced_vector is None:
            continue

        # Build enforced_values dict from tool_call fields + enforced vector
        original_args: Dict[str, Any] = exp.tool_call.get("arguments", {})
        field_names = list(original_args.keys())
        ev = exp.enforced_vector

        enforced_values: Dict[str, float] = {}
        for i, fname in enumerate(field_names):
            if i < len(ev):
                enforced_values[fname] = float(ev[i])

        corrected_tc = corrected_tool_call(exp.tool_call, enforced_values)

        # Identify changed fields
        corrected_fields = [
            k for k, v in enforced_values.items()
            if abs(float(original_args.get(k, v)) - v) > 1e-9
        ]

        # Build messages list with corrected assistant turn
        messages = list(exp.conversation_context) + [
            {"role": "assistant", "content": _tool_call_to_text(corrected_tc)}
        ]

        examples.append({
            "messages":            messages,
            "original_tool_call":  exp.tool_call,
            "enforcement_distance": exp.distance,
            "corrected_fields":    corrected_fields,
        })

    return examples


# ---------------------------------------------------------------------------
# DPO adapter
# ---------------------------------------------------------------------------


def to_dpo_pairs(
    experiences: List[EnforcementExperience],
    max_pairs: int = 1000,
) -> List[Dict[str, Any]]:
    """Convert APPROVE + REJECT pairs into DPO preference examples.

    Each example contains:

    - ``prompt``: conversation context as a message list.
    - ``chosen``: the APPROVE tool call text (preferred).
    - ``rejected``: the REJECT tool call text (dispreferred).
    - ``chosen_reward``: reward of the approved experience.
    - ``rejected_reward``: reward of the rejected experience.

    Pairs are matched on ``breaker_mode``; for each REJECT the nearest
    APPROVE by timestamp in the same mode is used.
    """
    approves = [e for e in experiences if e.result == "approve"]
    rejects  = [e for e in experiences if e.result == "reject"]

    pairs: List[Dict[str, Any]] = []
    for rej in rejects:
        candidates = [a for a in approves if a.breaker_mode == rej.breaker_mode]
        if not candidates:
            continue
        best = min(candidates, key=lambda a: abs(a.timestamp_ms - rej.timestamp_ms))
        pairs.append({
            "prompt":           list(rej.conversation_context),
            "chosen":           _tool_call_to_text(best.tool_call),
            "rejected":         _tool_call_to_text(rej.tool_call),
            "chosen_reward":    best.reward,
            "rejected_reward":  rej.reward,
        })
        if len(pairs) >= max_pairs:
            break

    return pairs


# ---------------------------------------------------------------------------
# PPO adapter
# ---------------------------------------------------------------------------


def to_ppo_episodes(
    experiences: List[EnforcementExperience],
) -> List[Dict[str, Any]]:
    """Convert all experiences into PPO-compatible episodes.

    Each episode contains:

    - ``query``: conversation context as a single string.
    - ``response``: the tool call text the model generated.
    - ``reward``: scalar reward from the reward shaper.
    - ``reward_components``: detailed reward breakdown.
    """
    return [
        {
            "query":             _context_to_string(exp.conversation_context),
            "response":          _tool_call_to_text(exp.tool_call),
            "reward":            exp.reward,
            "reward_components": exp.reward_components or {},
        }
        for exp in experiences
    ]


# ---------------------------------------------------------------------------
# Analytics export
# ---------------------------------------------------------------------------


def to_analytics_dataframe(
    experiences: List[EnforcementExperience],
) -> Dict[str, List]:
    """Convert experiences to a columnar dict suitable for pandas DataFrame.

    Columns
    -------
    experience_id, timestamp_ms, result, distance, reward, breaker_mode,
    overload_score, n_violations, top_violation_name, top_violation_magnitude,
    proposed_gpu_seconds, enforced_gpu_seconds,
    budget_gpu_remaining, budget_api_remaining, budget_mutation_remaining.
    """
    cols: Dict[str, List] = {
        "experience_id":             [],
        "timestamp_ms":              [],
        "result":                    [],
        "distance":                  [],
        "reward":                    [],
        "breaker_mode":              [],
        "overload_score":            [],
        "n_violations":              [],
        "top_violation_name":        [],
        "top_violation_magnitude":   [],
        "proposed_gpu_seconds":      [],
        "enforced_gpu_seconds":      [],
        "budget_gpu_remaining":      [],
        "budget_api_remaining":      [],
        "budget_mutation_remaining": [],
    }

    for exp in experiences:
        cols["experience_id"].append(exp.experience_id)
        cols["timestamp_ms"].append(exp.timestamp_ms)
        cols["result"].append(exp.result)
        cols["distance"].append(exp.distance)
        cols["reward"].append(exp.reward)
        cols["breaker_mode"].append(exp.breaker_mode)
        cols["overload_score"].append(exp.overload_score)
        cols["n_violations"].append(len(exp.violations))

        # Top violation
        if exp.violations:
            top = max(exp.violations, key=lambda x: x[1])
            cols["top_violation_name"].append(top[0])
            cols["top_violation_magnitude"].append(top[1])
        else:
            cols["top_violation_name"].append(None)
            cols["top_violation_magnitude"].append(0.0)

        # GPU seconds from tool_call arguments
        args = exp.tool_call.get("arguments", {})
        cols["proposed_gpu_seconds"].append(args.get("gpu_seconds", None))

        # Enforced GPU seconds from enforced_vector (same index as proposed)
        enf_gpu: Optional[float] = None
        if exp.enforced_vector is not None:
            field_names = list(args.keys())
            if "gpu_seconds" in field_names:
                idx = field_names.index("gpu_seconds")
                if idx < len(exp.enforced_vector):
                    enf_gpu = float(exp.enforced_vector[idx])
        cols["enforced_gpu_seconds"].append(enf_gpu)

        # Budget columns
        bud = exp.budget_remaining or {}
        cols["budget_gpu_remaining"].append(bud.get("gpu_shift", None))
        cols["budget_api_remaining"].append(bud.get("external_api_shift", None))
        cols["budget_mutation_remaining"].append(bud.get("mutation_shift", None))

    return cols
