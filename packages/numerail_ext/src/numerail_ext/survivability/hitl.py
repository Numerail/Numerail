"""Human-in-the-loop review trigger configuration.

Provides ``HumanReviewTriggers`` — a pure dataclass that declares which
enforcement events require human review — and three preset profiles:
ADVISORY, SUPERVISORY, and MANDATORY.

The trigger configuration is decoupled from the gateway implementation:
``HumanReviewTriggers`` only decides *whether* a review is needed and
*which mode* (BLOCKING vs. NOTIFY) applies.  The ``ApprovalGateway``
handles the routing and collection of decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from numerail.engine import EnforcementResult
from numerail.protocols import ReviewOutcome  # noqa: F401 — re-exported for convenience


class ReviewMode(str, Enum):
    """How the review interacts with execution flow.

    - BLOCKING: execution pauses until the human reviewer decides; the
      supervised governor will not commit the action until a decision arrives
      or the timeout expires (at which point the action is denied)
    - NOTIFY: the human reviewer is informed asynchronously; execution
      proceeds immediately without waiting for a decision
    """

    BLOCKING = "blocking"
    NOTIFY = "notify"


class HumanReviewProfile(str, Enum):
    """Three preset profiles for human review intensity.

    - ADVISORY: lightest touch — review only on critical events (safe-stop
      entry, audit-chain failure).  Suitable for low-stakes AI workloads.
    - SUPERVISORY: recommended default — review on rejections, significant
      corrections, and budget exhaustion.  Appropriate for most production
      AI governance deployments.
    - MANDATORY: full human oversight — review on everything except clean
      approvals.  Required for high-stakes or regulated environments where
      every non-trivial enforcement event must be visible to a human.
    """

    ADVISORY = "advisory"
    SUPERVISORY = "supervisory"
    MANDATORY = "mandatory"


# Priority order for highest_priority_trigger — lower index = higher priority.
_BLOCKING_PRIORITY = [
    "on_audit_chain_failure",
    "on_safe_stop_entry",
    "on_reject",
    "on_projection_forbidden_reject",
    "on_project_confirmation_required",
    "on_budget_exhaustion",
    "on_policy_version_change",
    "on_breaker_mode_change",
    "on_project_flagged",
    "on_budget_low",
    "on_trusted_field_mismatch",
]

_NOTIFY_PRIORITY = _BLOCKING_PRIORITY  # same order, lower tier


@dataclass
class HumanReviewTriggers:
    """Declarative configuration for human-in-the-loop review triggers.

    Each ``on_*`` field maps an enforcement event to a ``ReviewMode`` (or
    ``None`` to disable that trigger).  Threshold fields (floats) gate their
    companion ``*_mode`` fields.

    Use ``from_profile()`` to start from a sensible preset, or construct
    directly for full control.

    Parameters
    ----------
    on_reject : ReviewMode or None
        Trigger when the enforcement result is REJECT.
    on_project_confirmation_required : ReviewMode or None
        Trigger when the routing decision is ``confirmation``.
    on_project_flagged : ReviewMode or None
        Trigger when the routing decision is ``flagged``.
    on_projection_forbidden_reject : ReviewMode or None
        Trigger when projection is forbidden and the action is rejected.
    on_breaker_mode_change : ReviewMode or None
        Trigger when the circuit breaker transitions to a new mode.
    on_safe_stop_entry : ReviewMode or None
        Trigger when the breaker enters SAFE_STOP mode.
    on_budget_exhaustion : ReviewMode or None
        Trigger when any budget reaches zero remaining.
    on_budget_low_remaining_fraction : float or None
        Fraction threshold below which the budget-low trigger fires
        (e.g. 0.10 = fire when less than 10% remains).
    on_budget_low_mode : ReviewMode or None
        Review mode to apply when the budget-low threshold fires.
    on_policy_version_change : ReviewMode or None
        Trigger when a new policy version is activated.
    on_audit_chain_failure : ReviewMode or None
        Trigger when audit-chain verification fails (tamper-evident check).
    on_trusted_field_large_mismatch_fraction : float or None
        Fraction threshold for trusted-field mismatch magnitude (ratio of
        |provider_value - ai_value| / max(|ai_value|, 1)).  Fires when any
        field exceeds this fraction.
    on_trusted_field_mismatch_mode : ReviewMode or None
        Review mode to apply when the trusted-field mismatch threshold fires.
    review_timeout_seconds : float
        How long to wait for a human decision before expiring the review
        and denying the action (fail-closed default).
    max_escalation_depth : int
        Maximum number of times a review may be escalated before the action
        is automatically denied.
    """

    on_reject: Optional[ReviewMode] = None
    on_project_confirmation_required: Optional[ReviewMode] = None
    on_project_flagged: Optional[ReviewMode] = None
    on_projection_forbidden_reject: Optional[ReviewMode] = None
    on_breaker_mode_change: Optional[ReviewMode] = None
    on_safe_stop_entry: Optional[ReviewMode] = None
    on_budget_exhaustion: Optional[ReviewMode] = None
    on_budget_low_remaining_fraction: Optional[float] = None
    on_budget_low_mode: Optional[ReviewMode] = None
    on_policy_version_change: Optional[ReviewMode] = None
    on_audit_chain_failure: Optional[ReviewMode] = None
    on_trusted_field_large_mismatch_fraction: Optional[float] = None
    on_trusted_field_mismatch_mode: Optional[ReviewMode] = None
    review_timeout_seconds: float = 300.0
    max_escalation_depth: int = 2

    @classmethod
    def from_profile(cls, profile: HumanReviewProfile) -> "HumanReviewTriggers":
        """Construct a ``HumanReviewTriggers`` from a preset profile.

        Parameters
        ----------
        profile : HumanReviewProfile
            One of ADVISORY, SUPERVISORY, or MANDATORY.

        Returns
        -------
        HumanReviewTriggers
            Pre-configured trigger set for the chosen profile.
        """
        if profile == HumanReviewProfile.ADVISORY:
            return cls(
                on_safe_stop_entry=ReviewMode.BLOCKING,
                on_audit_chain_failure=ReviewMode.BLOCKING,
                on_budget_exhaustion=ReviewMode.NOTIFY,
                review_timeout_seconds=600.0,
                max_escalation_depth=1,
            )

        if profile == HumanReviewProfile.SUPERVISORY:
            return cls(
                on_reject=ReviewMode.BLOCKING,
                on_project_confirmation_required=ReviewMode.BLOCKING,
                on_projection_forbidden_reject=ReviewMode.BLOCKING,
                on_safe_stop_entry=ReviewMode.BLOCKING,
                on_budget_exhaustion=ReviewMode.BLOCKING,
                on_audit_chain_failure=ReviewMode.BLOCKING,
                on_breaker_mode_change=ReviewMode.NOTIFY,
                on_project_flagged=ReviewMode.NOTIFY,
                on_budget_low_remaining_fraction=0.10,
                on_budget_low_mode=ReviewMode.NOTIFY,
                review_timeout_seconds=300.0,
                max_escalation_depth=2,
            )

        if profile == HumanReviewProfile.MANDATORY:
            return cls(
                on_reject=ReviewMode.BLOCKING,
                on_project_confirmation_required=ReviewMode.BLOCKING,
                on_project_flagged=ReviewMode.BLOCKING,
                on_projection_forbidden_reject=ReviewMode.BLOCKING,
                on_breaker_mode_change=ReviewMode.BLOCKING,
                on_safe_stop_entry=ReviewMode.BLOCKING,
                on_budget_exhaustion=ReviewMode.BLOCKING,
                on_policy_version_change=ReviewMode.BLOCKING,
                on_audit_chain_failure=ReviewMode.BLOCKING,
                on_budget_low_remaining_fraction=0.20,
                on_budget_low_mode=ReviewMode.NOTIFY,
                review_timeout_seconds=120.0,
                max_escalation_depth=2,
            )

        raise ValueError(f"Unknown profile: {profile!r}")

    def evaluate_triggers(
        self,
        *,
        enforcement_result: Optional[str],
        routing_decision: Optional[str],
        breaker_mode_changed: bool,
        safe_stop_entered: bool,
        budget_exhausted: bool,
        budget_remaining_fraction: Optional[float],
        policy_version_changed: bool,
        audit_chain_failed: bool,
        trusted_field_mismatch_fraction: Optional[float],
    ) -> List[Tuple[str, ReviewMode]]:
        """Evaluate all triggers against the current enforcement context.

        This is a pure function — it reads the trigger configuration and the
        provided inputs; it does not modify any state.

        Parameters
        ----------
        enforcement_result : str or None
            The enforcement result string (``"approve"``, ``"project"``,
            ``"reject"``), or ``None`` if not applicable.
        routing_decision : str or None
            The routing decision label (``"flagged"``, ``"confirmation"``,
            ``"hard_reject"``, ``"silent"``), or ``None``.
        breaker_mode_changed : bool
            True if the circuit breaker just transitioned to a new mode.
        safe_stop_entered : bool
            True if the circuit breaker just entered SAFE_STOP mode.
        budget_exhausted : bool
            True if any tracked budget has reached zero.
        budget_remaining_fraction : float or None
            Lowest remaining-budget fraction across all tracked budgets,
            or ``None`` if no budgets are configured.
        policy_version_changed : bool
            True if a new policy version was activated in this cycle.
        audit_chain_failed : bool
            True if the audit-chain integrity check failed.
        trusted_field_mismatch_fraction : float or None
            Largest trusted-field mismatch fraction observed, or ``None``
            if no trusted fields are configured.

        Returns
        -------
        list of (trigger_name, ReviewMode)
            All triggers that fired, in evaluation order.  May be empty.
        """
        fired: List[Tuple[str, ReviewMode]] = []

        # Audit chain failure — checked first, highest severity
        if audit_chain_failed and self.on_audit_chain_failure is not None:
            fired.append(("on_audit_chain_failure", self.on_audit_chain_failure))

        # Safe-stop entry
        if safe_stop_entered and self.on_safe_stop_entry is not None:
            fired.append(("on_safe_stop_entry", self.on_safe_stop_entry))

        # Reject
        if enforcement_result == "reject" and self.on_reject is not None:
            fired.append(("on_reject", self.on_reject))

        # Projection forbidden → reject
        if (
            enforcement_result == "reject"
            and routing_decision == "hard_reject"
            and self.on_projection_forbidden_reject is not None
        ):
            fired.append(("on_projection_forbidden_reject", self.on_projection_forbidden_reject))

        # Confirmation required
        if (
            routing_decision == "confirmation"
            and self.on_project_confirmation_required is not None
        ):
            fired.append(("on_project_confirmation_required", self.on_project_confirmation_required))

        # Budget exhaustion
        if budget_exhausted and self.on_budget_exhaustion is not None:
            fired.append(("on_budget_exhaustion", self.on_budget_exhaustion))

        # Policy version change
        if policy_version_changed and self.on_policy_version_change is not None:
            fired.append(("on_policy_version_change", self.on_policy_version_change))

        # Breaker mode change (not SAFE_STOP — that's handled above)
        if breaker_mode_changed and not safe_stop_entered and self.on_breaker_mode_change is not None:
            fired.append(("on_breaker_mode_change", self.on_breaker_mode_change))

        # Flagged projection
        if routing_decision == "flagged" and self.on_project_flagged is not None:
            fired.append(("on_project_flagged", self.on_project_flagged))

        # Budget low threshold
        if (
            self.on_budget_low_remaining_fraction is not None
            and self.on_budget_low_mode is not None
            and budget_remaining_fraction is not None
            and budget_remaining_fraction < self.on_budget_low_remaining_fraction
        ):
            fired.append(("on_budget_low", self.on_budget_low_mode))

        # Trusted field mismatch
        if (
            self.on_trusted_field_large_mismatch_fraction is not None
            and self.on_trusted_field_mismatch_mode is not None
            and trusted_field_mismatch_fraction is not None
            and trusted_field_mismatch_fraction >= self.on_trusted_field_large_mismatch_fraction
        ):
            fired.append(("on_trusted_field_mismatch", self.on_trusted_field_mismatch_mode))

        return fired

    def highest_priority_trigger(
        self,
        fired_triggers: List[Tuple[str, ReviewMode]],
    ) -> Optional[Tuple[str, ReviewMode]]:
        """Return the highest-priority trigger from a fired list.

        Priority order: BLOCKING triggers first, then NOTIFY triggers, each
        group ordered by severity:

        audit_chain_failure > safe_stop_entry > reject >
        projection_forbidden > confirmation_required > budget_exhaustion >
        policy_version_change > breaker_mode_change > flagged >
        budget_low > trusted_field_mismatch

        Parameters
        ----------
        fired_triggers : list of (trigger_name, ReviewMode)
            The output of ``evaluate_triggers``.

        Returns
        -------
        (trigger_name, ReviewMode) or None
            The most important trigger, or ``None`` if the list is empty.
        """
        if not fired_triggers:
            return None

        def _priority(item: Tuple[str, ReviewMode]) -> Tuple[int, int]:
            name, mode = item
            tier = 0 if mode == ReviewMode.BLOCKING else 1
            try:
                rank = _BLOCKING_PRIORITY.index(name)
            except ValueError:
                rank = len(_BLOCKING_PRIORITY)
            return (tier, rank)

        return min(fired_triggers, key=_priority)
