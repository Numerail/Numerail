"""EnforcementRLOrchestrator — collect-train-evaluate cycle coordinator.

Coordinates enforcement experience collection, reward shaping,
training data export, and improvement tracking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from numerail_learn import adapter
from numerail_learn.experience import EnforcementExperience, EnforcementExperienceBuffer
from numerail_learn.reward import EnforcementRewardShaper


# ---------------------------------------------------------------------------
# Stats dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Statistics for a single collection episode."""

    total_actions: int
    approve_count: int
    project_count: int
    reject_count: int
    approval_rate: float
    mean_distance: float
    mean_reward: float
    breaker_transitions: int
    budget_exhaustion_events: int


@dataclass
class TrainingMetrics:
    """Metrics from a single training run."""

    method: str  # "sft", "dpo", "ppo"
    examples_used: int
    epochs: int
    final_loss: Optional[float]
    training_time_seconds: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EnforcementRLOrchestrator:
    """Coordinates enforcement experience collection and model training.

    This orchestrator does not run a live LLM agent — it provides the
    infrastructure for collecting experiences and producing training data.
    The actual agent integration and model training are deployment-specific.

    Parameters
    ----------
    governor : Any
        A ``StateTransitionGovernor`` instance used to pull budget/breaker
        state from. Only its ``backend.budget_remaining()`` method is called.
    buffer : EnforcementExperienceBuffer
        The experience buffer to write into.
    reward_shaper : EnforcementRewardShaper
        Reward shaper applied to every recorded step.
    schema_fields : Optional[List[str]]
        Field names in schema order, used for dimension feedback.
    budget_initial : Optional[Dict[str, float]]
        Initial budget values, used for budget bonus computation.
    """

    def __init__(
        self,
        governor: Any,
        buffer: EnforcementExperienceBuffer,
        reward_shaper: EnforcementRewardShaper,
        schema_fields: Optional[List[str]] = None,
        budget_initial: Optional[Dict[str, float]] = None,
    ) -> None:
        self._governor     = governor
        self._buffer       = buffer
        self._shaper       = reward_shaper
        self._schema_fields = schema_fields
        self._budget_initial = budget_initial

        self._approval_rate_history: List[Tuple[float, float]] = []
        self._mean_reward_history:   List[Tuple[float, float]] = []
        self._episode_count = 0

        # Track state within the current episode
        self._episode_start_size = 0
        self._last_breaker_mode: Optional[str] = None
        self._breaker_transitions = 0
        self._budget_exhaustion_events = 0

    # ── Recording ──────────────────────────────────────────────────────────

    def record_step(
        self,
        conversation_context: List[Dict[str, Any]],
        tool_call: Dict[str, Any],
        governed_step: Any,
        proposed_vector: Optional[np.ndarray] = None,
    ) -> EnforcementExperience:
        """Record a single GovernedStep as a training experience.

        Extracts enforcement output, breaker mode, and budget from the
        governed_step. Computes reward via the shaper. Stores in buffer.
        """
        nr = getattr(governed_step, "numerail_result", {}) or {}
        breaker = getattr(governed_step, "breaker", None)

        breaker_mode  = "closed"
        overload_score = 0.0
        if breaker is not None:
            mode_val = getattr(breaker, "mode", None)
            if mode_val is not None:
                breaker_mode = mode_val.value if hasattr(mode_val, "value") else str(mode_val)
            overload_score = float(getattr(breaker, "overload_score", 0.0))

        # Track breaker transitions within episode
        if self._last_breaker_mode is not None and breaker_mode != self._last_breaker_mode:
            self._breaker_transitions += 1
        self._last_breaker_mode = breaker_mode

        # Budget
        budget_remaining: Dict[str, float] = {}
        try:
            budget_remaining = dict(self._governor.backend.budget_remaining())
        except Exception:
            pass

        # Budget exhaustion: any budget <= 0
        if any(v <= 0.0 for v in budget_remaining.values()):
            self._budget_exhaustion_events += 1

        # Policy digest
        policy_digest = str(nr.get("policy_digest", ""))

        # Proposed vector: fallback to extracting from tool_call
        if proposed_vector is None:
            args = tool_call.get("arguments", {})
            proposed_vector = np.array(list(args.values()), dtype=np.float64)

        # Record into buffer (unshaped first)
        exp = self._buffer.record(
            conversation_context=conversation_context,
            tool_call=tool_call,
            proposed_vector=proposed_vector,
            enforcement_output=governed_step,
            breaker_mode=breaker_mode,
            budget_remaining=budget_remaining,
            overload_score=overload_score,
            policy_digest=policy_digest,
        )

        # Apply reward shaping and replace in buffer
        shaped = self._shaper.shape_experience(
            exp,
            schema_fields=self._schema_fields,
            budget_initial=self._budget_initial,
        )

        # Swap the experience in the buffer with the shaped version
        with self._buffer._lock:
            for i, item in enumerate(self._buffer._buffer):
                if item.experience_id == exp.experience_id:
                    self._buffer._buffer[i] = shaped
                    break

        return shaped

    # ── Episode boundary ───────────────────────────────────────────────────

    def record_episode_boundary(self) -> EpisodeStats:
        """Mark end of episode, snapshot metrics, return EpisodeStats."""
        all_exps = self._buffer.get_all()
        episode_size = len(all_exps) - self._episode_start_size

        # Only stats for this episode
        episode_exps = all_exps[-episode_size:] if episode_size > 0 else []

        approve = sum(1 for e in episode_exps if e.result == "approve")
        project = sum(1 for e in episode_exps if e.result == "project")
        reject  = sum(1 for e in episode_exps if e.result == "reject")
        total   = len(episode_exps)

        approval_rate = approve / total if total > 0 else 0.0

        distances = [e.distance for e in episode_exps if e.distance >= 0]
        mean_dist = float(np.mean(distances)) if distances else 0.0

        rewards     = [e.reward for e in episode_exps]
        mean_reward = float(np.mean(rewards)) if rewards else 0.0

        ts = float(time.time_ns() // 1_000_000)
        self._approval_rate_history.append((ts, approval_rate))
        self._mean_reward_history.append((ts, mean_reward))
        self._episode_count += 1

        stats = EpisodeStats(
            total_actions=total,
            approve_count=approve,
            project_count=project,
            reject_count=reject,
            approval_rate=approval_rate,
            mean_distance=mean_dist,
            mean_reward=mean_reward,
            breaker_transitions=self._breaker_transitions,
            budget_exhaustion_events=self._budget_exhaustion_events,
        )

        # Reset episode-level trackers
        self._episode_start_size = len(all_exps)
        self._last_breaker_mode  = None
        self._breaker_transitions = 0
        self._budget_exhaustion_events = 0

        return stats

    # ── Export ─────────────────────────────────────────────────────────────

    def export_sft_data(self) -> List[Dict[str, Any]]:
        """Export PROJECT experiences as SFT training examples."""
        return adapter.to_sft_examples(self._buffer.get_project_experiences())

    def export_dpo_data(self, max_pairs: int = 1000) -> List[Dict[str, Any]]:
        """Export APPROVE/REJECT pairs as DPO training examples."""
        return adapter.to_dpo_pairs(self._buffer.get_all(), max_pairs=max_pairs)

    def export_ppo_data(self) -> List[Dict[str, Any]]:
        """Export all experiences as PPO episodes."""
        return adapter.to_ppo_episodes(self._buffer.get_all())

    def export_analytics(self) -> Dict[str, List]:
        """Export all experiences as columnar analytics data."""
        return adapter.to_analytics_dataframe(self._buffer.get_all())

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def approval_rate(self) -> float:
        return self._buffer.approval_rate

    @property
    def approval_rate_history(self) -> List[Tuple[float, float]]:
        return list(self._approval_rate_history)

    # ── Reports ────────────────────────────────────────────────────────────

    def dimension_report(self) -> Dict[str, Any]:
        """Which dimensions the model most frequently violates.

        Returns
        -------
        dict with:
            ``violation_frequency``, ``mean_overshoot``,
            ``most_violated``, ``recommendation``.
        """
        freq = self._buffer.dimension_violation_frequency()

        # Mean overshoot: mean positive delta per dimension from PROJECT experiences
        overshoot: Dict[str, List[float]] = {}
        for exp in self._buffer.get_project_experiences():
            if exp.dimension_feedback:
                for dim, delta in exp.dimension_feedback.items():
                    if delta > 0:
                        overshoot.setdefault(dim, []).append(delta)
        mean_overshoot = {k: float(np.mean(v)) for k, v in overshoot.items()}

        most_violated = max(freq, key=freq.get) if freq else ""
        recommendation = ""
        if most_violated:
            mo = mean_overshoot.get(most_violated, 0.0)
            recommendation = (
                f"Dimension '{most_violated}' was violated {freq[most_violated]} times "
                f"with mean overshoot {mo:.4f}. Consider reducing proposals for this dimension."
            )

        return {
            "violation_frequency": freq,
            "mean_overshoot":      mean_overshoot,
            "most_violated":       most_violated,
            "recommendation":      recommendation,
        }

    def improvement_report(self) -> Dict[str, Any]:
        """Summary of model improvement over time."""
        history = [r for _, r in self._approval_rate_history]
        rewards  = [r for _, r in self._mean_reward_history]

        initial = history[0]  if history else 0.0
        current = history[-1] if history else 0.0

        all_exps = self._buffer.get_all()
        return {
            "total_experiences":       len(all_exps),
            "episodes_completed":      self._episode_count,
            "initial_approval_rate":   initial,
            "current_approval_rate":   current,
            "approval_rate_trend":     list(history),
            "mean_reward_trend":       list(rewards),
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Save buffer and metadata to disk."""
        buf_path  = path + ".buffer.json"
        meta_path = path + ".meta.json"

        self._buffer.export_json(buf_path)

        meta = {
            "episode_count":            self._episode_count,
            "episode_start_size":       self._episode_start_size,
            "approval_rate_history":    self._approval_rate_history,
            "mean_reward_history":      self._mean_reward_history,
            "last_breaker_mode":        self._last_breaker_mode,
            "breaker_transitions":      self._breaker_transitions,
            "budget_exhaustion_events": self._budget_exhaustion_events,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    def load_state(self, path: str) -> None:
        """Load buffer and metadata from disk."""
        buf_path  = path + ".buffer.json"
        meta_path = path + ".meta.json"

        self._buffer.clear()
        self._buffer.import_json(buf_path)

        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._episode_count             = meta.get("episode_count", 0)
        self._episode_start_size        = meta.get("episode_start_size", 0)
        self._approval_rate_history     = [tuple(x) for x in meta.get("approval_rate_history", [])]
        self._mean_reward_history       = [tuple(x) for x in meta.get("mean_reward_history", [])]
        self._last_breaker_mode         = meta.get("last_breaker_mode")
        self._breaker_transitions       = meta.get("breaker_transitions", 0)
        self._budget_exhaustion_events  = meta.get("budget_exhaustion_events", 0)
