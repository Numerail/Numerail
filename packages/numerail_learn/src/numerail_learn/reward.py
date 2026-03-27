"""Reward shaping from Numerail enforcement decisions.

Three preset configurations are provided:
- ``conservative_shaper()``: high penalties, trains caution
- ``permissive_shaper()``:  low penalties, rewards boundary exploration
- ``strict_shaper()``:      maximum penalty for any non-APPROVE
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from numerail_learn.experience import EnforcementExperience


class EnforcementRewardShaper:
    """Compute RL training rewards from Numerail enforcement decisions.

    Three additive components:

    - **Approval component**: ``+approve_reward`` / ``+project_base`` /
      ``reject_penalty`` depending on result.
    - **Distance component**: ``-distance_scale * distance``
      (0 for APPROVE).
    - **Violation component**: ``-violation_scale * sum(magnitudes)``.

    Optional bonus:

    - **Budget efficiency**: ``+budget_bonus_scale * mean_remaining_fraction``
      rewards conserving budget.

    Parameters
    ----------
    approve_reward :
        Reward added for APPROVE decisions.
    project_base :
        Base reward for PROJECT decisions (before distance/violation penalties).
    reject_penalty :
        Reward added for REJECT decisions (typically negative).
    distance_scale :
        Penalty per unit of enforcement distance.
    violation_scale :
        Penalty per unit of total violation magnitude.
    budget_bonus_scale :
        Bonus per unit of mean budget-remaining fraction.
    """

    def __init__(
        self,
        approve_reward: float = 1.0,
        project_base: float = 0.0,
        reject_penalty: float = -1.0,
        distance_scale: float = 0.1,
        violation_scale: float = 0.5,
        budget_bonus_scale: float = 0.05,
    ) -> None:
        self.approve_reward    = approve_reward
        self.project_base      = project_base
        self.reject_penalty    = reject_penalty
        self.distance_scale    = distance_scale
        self.violation_scale   = violation_scale
        self.budget_bonus_scale = budget_bonus_scale

    # ── Core compute ───────────────────────────────────────────────────────

    def compute_reward(
        self,
        result: str,
        distance: float,
        violations: List[Tuple[str, float]],
        budget_remaining: Optional[Dict[str, float]] = None,
        budget_initial: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute scalar reward from enforcement outcome."""
        return self.compute_detailed_reward(
            result=result,
            distance=distance,
            violations=violations,
            budget_remaining=budget_remaining,
            budget_initial=budget_initial,
        )["total_reward"]

    def compute_detailed_reward(
        self,
        result: str,
        distance: float,
        violations: List[Tuple[str, float]],
        proposed_vector: Optional[np.ndarray] = None,
        enforced_vector: Optional[np.ndarray] = None,
        schema_fields: Optional[List[str]] = None,
        budget_remaining: Optional[Dict[str, float]] = None,
        budget_initial: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute detailed reward breakdown.

        Returns
        -------
        dict with keys:
            ``total_reward``, ``approval_component``, ``distance_component``,
            ``violation_component``, ``budget_component``, ``dimension_feedback``.
        """
        r = result.lower()

        # Approval component
        if r == "approve":
            approval_component = self.approve_reward
        elif r == "project":
            approval_component = self.project_base
        else:
            approval_component = self.reject_penalty

        # Distance component (0 for approve; -1.0 sentinel for reject → 0 penalty)
        effective_distance = max(0.0, distance)  # treat -1 as 0 for scaling
        distance_component = -self.distance_scale * effective_distance

        # Violation component
        total_viol = sum(mag for _, mag in violations)
        violation_component = -self.violation_scale * total_viol

        # Budget bonus
        budget_component = 0.0
        if self.budget_bonus_scale > 0 and budget_remaining and budget_initial:
            fractions = []
            for k, init in budget_initial.items():
                if init > 0 and k in budget_remaining:
                    fractions.append(max(0.0, budget_remaining[k] / init))
            if fractions:
                budget_component = self.budget_bonus_scale * (sum(fractions) / len(fractions))

        total_reward = approval_component + distance_component + violation_component + budget_component

        # Dimension feedback: only for PROJECT with full vector info
        dimension_feedback: Dict[str, float] = {}
        if (r == "project"
                and proposed_vector is not None
                and enforced_vector is not None
                and schema_fields is not None):
            pv = np.asarray(proposed_vector, dtype=np.float64)
            ev = np.asarray(enforced_vector, dtype=np.float64)
            n  = min(len(pv), len(ev), len(schema_fields))
            for i in range(n):
                delta = float(pv[i] - ev[i])
                if abs(delta) > 1e-9:
                    dimension_feedback[schema_fields[i]] = round(delta, 6)

        return {
            "total_reward":        round(total_reward, 6),
            "approval_component":  round(approval_component, 6),
            "distance_component":  round(distance_component, 6),
            "violation_component": round(violation_component, 6),
            "budget_component":    round(budget_component, 6),
            "dimension_feedback":  dimension_feedback,
        }

    # ── Experience shaping ─────────────────────────────────────────────────

    def shape_experience(
        self,
        experience: EnforcementExperience,
        schema_fields: Optional[List[str]] = None,
        budget_initial: Optional[Dict[str, float]] = None,
    ) -> EnforcementExperience:
        """Return a new experience with reward and dimension_feedback populated.

        Does not mutate the input experience.
        """
        detail = self.compute_detailed_reward(
            result=experience.result,
            distance=experience.distance,
            violations=experience.violations,
            proposed_vector=experience.proposed_vector,
            enforced_vector=experience.enforced_vector,
            schema_fields=schema_fields,
            budget_remaining=experience.budget_remaining if experience.budget_remaining else None,
            budget_initial=budget_initial,
        )

        components = {
            "approval_component":  detail["approval_component"],
            "distance_component":  detail["distance_component"],
            "violation_component": detail["violation_component"],
            "budget_component":    detail["budget_component"],
        }

        return EnforcementExperience(
            experience_id=       experience.experience_id,
            action_id=           experience.action_id,
            timestamp_ms=        experience.timestamp_ms,
            conversation_context=experience.conversation_context,
            tool_call=           experience.tool_call,
            proposed_vector=     experience.proposed_vector,
            result=              experience.result,
            enforced_vector=     experience.enforced_vector,
            distance=            experience.distance,
            violations=          experience.violations,
            solver_method=       experience.solver_method,
            routing_decision=    experience.routing_decision,
            breaker_mode=        experience.breaker_mode,
            budget_remaining=    experience.budget_remaining,
            overload_score=      experience.overload_score,
            policy_digest=       experience.policy_digest,
            reward=              detail["total_reward"],
            reward_components=   components,
            dimension_feedback=  detail["dimension_feedback"] or None,
        )


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------


def conservative_shaper() -> EnforcementRewardShaper:
    """High penalties — trains models to strongly avoid constraint boundaries."""
    return EnforcementRewardShaper(
        approve_reward=1.0,
        project_base=-0.2,
        reject_penalty=-2.0,
        distance_scale=0.2,
        violation_scale=1.0,
        budget_bonus_scale=0.1,
    )


def permissive_shaper() -> EnforcementRewardShaper:
    """Small penalties — trains models to use available authority efficiently."""
    return EnforcementRewardShaper(
        approve_reward=1.0,
        project_base=0.3,
        reject_penalty=-0.5,
        distance_scale=0.05,
        violation_scale=0.2,
        budget_bonus_scale=0.02,
    )


def strict_shaper() -> EnforcementRewardShaper:
    """Maximum penalty — trains models to always stay well within bounds."""
    return EnforcementRewardShaper(
        approve_reward=1.0,
        project_base=-1.0,
        reject_penalty=-3.0,
        distance_scale=0.5,
        violation_scale=2.0,
        budget_bonus_scale=0.0,
    )
