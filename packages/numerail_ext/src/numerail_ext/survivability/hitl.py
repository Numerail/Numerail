"""Human-in-the-loop review trigger configuration and supervised governor.

Provides:
- ``HumanReviewTriggers`` — pure dataclass declaring which enforcement events
  require human review, with three preset profiles (ADVISORY, SUPERVISORY,
  MANDATORY).
- ``PendingAction`` — record of an enforcement action awaiting human review.
- ``SupervisedStepResult`` — complete result of one supervised governance cycle.
- ``SupervisedGovernor`` — wraps ``StateTransitionGovernor`` with a HITL gate,
  TOCTOU re-enforcement, escalation/defer logic, and SHA-256 audit chaining.

The trigger configuration is decoupled from the gateway implementation:
``HumanReviewTriggers`` only decides *whether* a review is needed and
*which mode* (BLOCKING vs. NOTIFY) applies.  The ``ApprovalGateway``
handles the routing and collection of decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from numerail.engine import EnforcementResult
from numerail.protocols import (
    ApprovalGateway,  # noqa: F401
    HumanDecision,
    HumanDecisionAction,
    ReviewOutcome,  # noqa: F401 — re-exported for convenience
)


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


# ── Stage 2: SupervisedGovernor types and implementation ─────────────────────


@dataclass
class PendingAction:
    """Record of an enforcement action awaiting human review.

    Created by ``SupervisedGovernor.step()`` when a BLOCKING trigger fires.
    Consumed by ``SupervisedGovernor.resolve_pending()`` when a human decision
    arrives or the review timeout expires.

    Attributes
    ----------
    review_id : str
        The review identifier returned by the gateway.
    enforcement_output : GovernedStep
        The governed step awaiting review.
    submitted_at_ms : float
        Wall-clock submission time in milliseconds (for timeout tracking).
    trigger_reason : str
        The trigger name that caused this review.
    context : dict
        Operational context forwarded to the reviewer.
    review_mode : ReviewMode
        Always BLOCKING for PendingAction records (NOTIFY never creates one).
    escalation_depth : int
        Number of escalations so far (0 = not yet escalated).
    deferred : bool
        Whether the reviewer has already deferred once (max one deferral).
    """

    review_id: str
    enforcement_output: Any          # GovernedStep in practice
    submitted_at_ms: float
    trigger_reason: str
    context: Dict[str, Any]
    review_mode: ReviewMode
    escalation_depth: int = 0
    deferred: bool = False


@dataclass
class SupervisedStepResult:
    """Complete result of one supervised governance cycle.

    Attributes
    ----------
    outcome : ReviewOutcome
        EXECUTED, PENDING_REVIEW, DENIED, or EXPIRED.
    enforcement_output : GovernedStep
        Raw output from the inner ``StateTransitionGovernor``.
    human_decision : HumanDecision or None
        The reviewer's decision, if one was made in this cycle.
    re_enforcement_output : Mapping[str, Any] or None
        TOCTOU re-enforcement result after human approval or MODIFY.
    trigger_reason : str or None
        Highest-priority trigger name that caused a review, or None.
    guidance_constraints : dict or None
        Advisory bounds from a DENY decision: field_name → (min, max).
    """

    outcome: ReviewOutcome
    enforcement_output: Any          # GovernedStep
    human_decision: Optional[HumanDecision] = None
    re_enforcement_output: Optional[Any] = None
    trigger_reason: Optional[str] = None
    guidance_constraints: Optional[Dict[str, Tuple[float, float]]] = None


# ── Internal SHA-256 audit chain for HITL events ──────────────────────────────


class _HitlAuditChain:
    """Internal SHA-256 hash-linked audit chain for HITL-specific events.

    Records human decisions and review expiry events in a tamper-evident
    sequence.  Separate from the engine's ``AuditChain`` (which only accepts
    ``EnforcementOutput``); this chain covers the HITL governance layer.

    The genesis hash is 64 zero hex digits.  Each record's ``chain_hash``
    is SHA-256(json.dumps({...record..., "prev_hash": prev}, sort_keys=True)).
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._prev_hash: str = "0" * 64  # genesis

    def append(self, record: Dict[str, Any]) -> str:
        """Append a record and return the new chain hash."""
        payload: Dict[str, Any] = {**record, "prev_hash": self._prev_hash}
        serialized = json.dumps(payload, sort_keys=True, default=str)
        new_hash = hashlib.sha256(serialized.encode()).hexdigest()
        self._records.append({**payload, "chain_hash": new_hash})
        self._prev_hash = new_hash
        return new_hash

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    @property
    def head_hash(self) -> str:
        return self._prev_hash


# ── Audit helper functions ────────────────────────────────────────────────────


def record_human_decision(
    audit_chain: _HitlAuditChain,
    human_decision: HumanDecision,
    enforcement_output: Any,
    re_enforcement_output: Optional[Any] = None,
) -> str:
    """Append a human-decision event to a HITL audit chain.

    Parameters
    ----------
    audit_chain : _HitlAuditChain
        The chain to append to.
    human_decision : HumanDecision
        The reviewer's decision.
    enforcement_output : GovernedStep
        The governed step that was reviewed.
    re_enforcement_output : Mapping[str, Any] or None
        TOCTOU re-enforcement result, if applicable.

    Returns
    -------
    str
        The new chain hash after appending.
    """
    enforcement_decision = None
    if hasattr(enforcement_output, "numerail_result"):
        enforcement_decision = enforcement_output.numerail_result.get("decision")

    re_decision = None
    if re_enforcement_output is not None:
        re_decision = (
            re_enforcement_output.get("decision")
            if hasattr(re_enforcement_output, "get")
            else None
        )

    record: Dict[str, Any] = {
        "event_type": "human_decision",
        "review_id": human_decision.review_id,
        "action": human_decision.action.value,
        "reviewer_id": human_decision.reviewer_id,
        "authenticated": human_decision.authenticated,
        "timestamp_ms": human_decision.timestamp_ms,
        "reason": human_decision.reason,
        "escalation_depth": human_decision.escalation_depth,
        "enforcement_decision": enforcement_decision,
        "re_enforcement_decision": re_decision,
    }
    return audit_chain.append(record)


def record_review_expiry(
    audit_chain: _HitlAuditChain,
    review_id: str,
    submitted_at_ms: float,
    expired_at_ms: float,
    trigger_reason: str,
) -> str:
    """Append a review-expiry event to a HITL audit chain.

    Parameters
    ----------
    audit_chain : _HitlAuditChain
        The chain to append to.
    review_id : str
        The review that expired.
    submitted_at_ms : float
        Submission timestamp in milliseconds.
    expired_at_ms : float
        Expiry timestamp in milliseconds.
    trigger_reason : str
        The trigger name that originally caused the review.

    Returns
    -------
    str
        The new chain hash after appending.
    """
    record: Dict[str, Any] = {
        "event_type": "review_expiry",
        "review_id": review_id,
        "submitted_at_ms": submitted_at_ms,
        "expired_at_ms": expired_at_ms,
        "elapsed_ms": expired_at_ms - submitted_at_ms,
        "trigger_reason": trigger_reason,
    }
    return audit_chain.append(record)


# ── SupervisedGovernor ────────────────────────────────────────────────────────


class SupervisedGovernor:
    """HITL-gated supervisory governor wrapping ``StateTransitionGovernor``.

    Adds a human-in-the-loop gate between enforcement and execution.  When a
    configured ``HumanReviewTriggers`` fires on an enforcement event:

    - **NOTIFY**: reviewer is notified asynchronously; execution proceeds
      immediately (EXECUTED or DENIED depending on the enforcement result).
    - **BLOCKING**: execution is paused; the action enters ``PENDING_REVIEW``
      until a human decides or the timeout expires (fail-closed → DENIED).

    Decision handling (``resolve_pending``):

    - **APPROVE**: TOCTOU re-enforces the original enforced vector against the
      *current* geometry before granting EXECUTED.
    - **DENY**: DENIED with optional advisory ``guidance_constraints``.
    - **MODIFY**: reviewer-supplied vector is re-enforced; EXECUTED if feasible,
      DENIED otherwise.
    - **ESCALATE**: re-submitted to the gateway up to ``max_escalation_depth``
      times; ceiling exceeded → DENIED.
    - **DEFER**: timeout extended once (reviewer gets a second full window);
      second defer → DENIED.

    Unauthenticated decisions (``HumanDecision.authenticated is False``) are
    rejected immediately and produce DENIED.

    All human decisions and expiry events are appended to an internal
    SHA-256 hash-linked audit chain (``hitl_audit_records``).

    Parameters
    ----------
    governor : StateTransitionGovernor
        Inner governor to delegate enforcement to.
    gateway : ApprovalGateway
        Human review gateway (use ``LocalApprovalGateway`` for testing).
    triggers : HumanReviewTriggers
        Trigger configuration controlling when reviews are requested.
    trusted_context_provider : TrustedContextProvider or None
        Optional provider for computing trusted-field mismatch fractions.
    _time_ms_fn : callable or None
        Injectable time source (``() -> float``) for deterministic testing.
        Defaults to ``time.time() * 1000``.
    """

    def __init__(
        self,
        governor: Any,                    # StateTransitionGovernor
        gateway: Any,                     # ApprovalGateway
        triggers: HumanReviewTriggers,
        trusted_context_provider: Optional[Any] = None,
        _time_ms_fn: Optional[Any] = None,
    ) -> None:
        self._governor = governor
        self._gateway = gateway
        self._triggers = triggers
        self._trusted_context_provider = trusted_context_provider
        self._time_ms_fn = _time_ms_fn
        self._pending: Dict[str, PendingAction] = {}
        self._hitl_audit = _HitlAuditChain()
        self._action_counter: int = 0
        self._last_breaker_mode: Optional[Any] = None   # BreakerMode
        self._last_policy_id: Optional[str] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_current_time_ms(self) -> float:
        """Return current wall-clock time in milliseconds."""
        if self._time_ms_fn is not None:
            return self._time_ms_fn()
        return time.time() * 1000.0

    def _detect_events(
        self,
        governed_step: Any,              # GovernedStep
        pre_mode: Optional[Any],         # BreakerMode before step
        post_mode: Any,                  # BreakerMode after step
        budget_state: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Extract trigger-relevant events from a governed step."""
        from .types import BreakerMode

        result_dict = governed_step.numerail_result
        enforcement_result: Optional[str] = result_dict.get("decision")
        routing_decision: Optional[str] = result_dict.get("routing")

        # Breaker events
        breaker_mode_changed = pre_mode is not None and pre_mode != post_mode
        safe_stop_entered = (
            post_mode == BreakerMode.SAFE_STOP
            and pre_mode != BreakerMode.SAFE_STOP
        )

        # Budget events
        budget_exhausted = False
        budget_remaining_fraction: Optional[float] = None
        if budget_state:
            values = [v for v in budget_state.values() if isinstance(v, (int, float))]
            if values:
                budget_exhausted = any(v <= 0 for v in values)
                envelope = governed_step.envelope
                ceiling_map = {
                    "gpu_shift": float(envelope.remaining_gpu_shift),
                    "external_api_shift": float(envelope.remaining_external_api_shift),
                    "mutation_shift": float(envelope.remaining_mutation_shift),
                }
                fractions = []
                for k, v in budget_state.items():
                    ceiling = ceiling_map.get(k, 0.0)
                    if ceiling > 0:
                        fractions.append(float(v) / ceiling)
                if fractions:
                    budget_remaining_fraction = min(fractions)

        # Policy version change
        current_policy_id = governed_step.envelope.policy_id
        policy_version_changed = (
            self._last_policy_id is not None
            and self._last_policy_id != current_policy_id
        )

        # Audit chain failure: proxy — audit_hash missing on a non-OPEN, non-reject path
        audit_chain_failed = (
            result_dict.get("audit_hash") is None
            and enforcement_result not in (None, "reject")
            and post_mode != BreakerMode.OPEN
        )

        # Trusted field mismatch
        trusted_field_mismatch_fraction: Optional[float] = None
        if self._trusted_context_provider is not None:
            try:
                provider_vals = self._trusted_context_provider.get_trusted_context()
                envelope_ctx = governed_step.envelope.trusted_context()
                mismatches = []
                for field_name, provider_val in provider_vals.items():
                    if field_name in envelope_ctx:
                        ai_val = envelope_ctx[field_name]
                        denom = max(abs(float(ai_val)), 1.0)
                        mismatches.append(abs(float(provider_val) - float(ai_val)) / denom)
                if mismatches:
                    trusted_field_mismatch_fraction = max(mismatches)
            except Exception:
                pass

        return {
            "enforcement_result": enforcement_result,
            "routing_decision": routing_decision,
            "breaker_mode_changed": breaker_mode_changed,
            "safe_stop_entered": safe_stop_entered,
            "budget_exhausted": budget_exhausted,
            "budget_remaining_fraction": budget_remaining_fraction,
            "policy_version_changed": policy_version_changed,
            "audit_chain_failed": audit_chain_failed,
            "trusted_field_mismatch_fraction": trusted_field_mismatch_fraction,
        }

    def _toctou_reenforce(
        self,
        governed_step: Any,                          # GovernedStep
        proposed_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Re-enforce against the *current* backend geometry (TOCTOU protection).

        If ``proposed_values`` is None, re-enforces the original enforced vector.
        The backend's currently-loaded config may differ from when the original
        step was taken (e.g. if another step ran, or a test injected a tighter
        config via ``set_active_config``).
        """
        if proposed_values is None:
            enforced = governed_step.numerail_result.get("enforced_values")
            if enforced is None:
                return {
                    "decision": "reject",
                    "enforced_values": None,
                    "feedback": {"message": "no enforced values available for TOCTOU check"},
                    "audit_hash": None,
                }
            proposed_values = dict(enforced)

        action_id = f"toctou-{uuid.uuid4().hex[:8]}"
        try:
            result = self._governor.backend.enforce(
                policy_id=governed_step.envelope.policy_id,
                proposed_action=proposed_values,
                action_id=action_id,
                trusted_context=governed_step.envelope.trusted_context(),
                execution_topic="hitl_toctou",
            )
            return dict(result)
        except Exception as exc:
            return {
                "decision": "reject",
                "enforced_values": None,
                "feedback": {"message": f"TOCTOU re-enforcement error: {exc}"},
                "audit_hash": None,
            }

    # ── Primary API ───────────────────────────────────────────────────────

    def step(
        self,
        *,
        request: Any,                     # WorkloadRequest
        snapshot: Any,                    # TelemetrySnapshot
        action_id: Optional[str] = None,
    ) -> SupervisedStepResult:
        """Execute one enforcement cycle with HITL gating.

        Calls the inner governor's ``enforce_next_step``, evaluates configured
        triggers against the result, and returns one of:

        - **EXECUTED** — no BLOCKING trigger fired (NOTIFY still notifies
          asynchronously), or enforcement passed without any trigger.
        - **DENIED** — enforcement rejected and no review configured.
        - **PENDING_REVIEW** — a BLOCKING trigger fired; use ``resolve_pending``
          to process the reviewer's decision.

        Parameters
        ----------
        request : WorkloadRequest
            The model's proposed workload authority.
        snapshot : TelemetrySnapshot
            Trusted current telemetry.
        action_id : str or None
            Optional identifier; auto-generated as ``"supervised-N"`` if absent.

        Returns
        -------
        SupervisedStepResult
        """
        if action_id is None:
            self._action_counter += 1
            action_id = f"supervised-{self._action_counter}"

        pre_mode = self._last_breaker_mode

        governed_step = self._governor.enforce_next_step(
            request=request,
            snapshot=snapshot,
            action_id=action_id,
        )

        post_mode = governed_step.breaker.mode
        self._last_breaker_mode = post_mode

        # Gather budget state for event detection
        try:
            budget_state: Optional[Dict[str, float]] = dict(
                self._governor.backend.budget_remaining()
            )
        except Exception:
            budget_state = None

        events = self._detect_events(governed_step, pre_mode, post_mode, budget_state)

        fired = self._triggers.evaluate_triggers(**events)
        top = self._triggers.highest_priority_trigger(fired)

        enforcement_result = governed_step.numerail_result.get("decision")

        # Update policy-id tracking for next step
        self._last_policy_id = governed_step.envelope.policy_id

        if top is None:
            # No trigger — pass through enforcement result directly
            if enforcement_result in ("approve", "project"):
                return SupervisedStepResult(
                    outcome=ReviewOutcome.EXECUTED,
                    enforcement_output=governed_step,
                )
            return SupervisedStepResult(
                outcome=ReviewOutcome.DENIED,
                enforcement_output=governed_step,
            )

        trigger_name, trigger_mode = top
        context: Dict[str, Any] = {
            "breaker_mode": (
                post_mode.value if hasattr(post_mode, "value") else str(post_mode)
            ),
            "enforcement_result": enforcement_result,
            "action_id": action_id,
        }

        if trigger_mode == ReviewMode.NOTIFY:
            # Fire-and-forget — notify reviewer but do not block execution
            self._gateway.submit_for_review(governed_step, trigger_name, context)
            if enforcement_result in ("approve", "project"):
                return SupervisedStepResult(
                    outcome=ReviewOutcome.EXECUTED,
                    enforcement_output=governed_step,
                    trigger_reason=trigger_name,
                )
            return SupervisedStepResult(
                outcome=ReviewOutcome.DENIED,
                enforcement_output=governed_step,
                trigger_reason=trigger_name,
            )

        # BLOCKING — submit for review and return PENDING_REVIEW
        review_id = self._gateway.submit_for_review(governed_step, trigger_name, context)
        submitted_at_ms = self._get_current_time_ms()

        self._pending[review_id] = PendingAction(
            review_id=review_id,
            enforcement_output=governed_step,
            submitted_at_ms=submitted_at_ms,
            trigger_reason=trigger_name,
            context=context,
            review_mode=trigger_mode,
            escalation_depth=0,
            deferred=False,
        )

        return SupervisedStepResult(
            outcome=ReviewOutcome.PENDING_REVIEW,
            enforcement_output=governed_step,
            trigger_reason=trigger_name,
        )

    def resolve_pending(self, review_id: str) -> Optional[SupervisedStepResult]:
        """Poll for a human decision on a pending review.

        Checks for timeout first (fail-closed → EXPIRED), then polls the
        gateway for a decision.  Returns ``None`` if still pending.

        Decision handling:

        - **APPROVE**: TOCTOU re-enforces; EXECUTED if feasible, DENIED if the
          geometry has tightened since the original step.
        - **DENY**: DENIED with optional ``guidance_constraints``.
        - **MODIFY**: reviewer's vector re-enforced; EXECUTED or DENIED.
        - **ESCALATE**: re-submits to gateway (new review_id); returns ``None``
          so the caller can poll the new id.  Ceiling exceeded → DENIED.
        - **DEFER**: extends timeout by one full window (once only); second
          defer → DENIED.
        - **Unauthenticated decision**: DENIED immediately.

        Parameters
        ----------
        review_id : str
            The review_id from a previous ``step()`` call.

        Returns
        -------
        SupervisedStepResult or None
        """
        pending = self._pending.get(review_id)
        if pending is None:
            return None

        now_ms = self._get_current_time_ms()
        timeout_ms = self._triggers.review_timeout_seconds * 1000.0
        deadline_ms = pending.submitted_at_ms + timeout_ms
        if pending.deferred:
            deadline_ms += timeout_ms  # deferred extends by one full timeout

        if now_ms > deadline_ms:
            del self._pending[review_id]
            record_review_expiry(
                self._hitl_audit,
                review_id=review_id,
                submitted_at_ms=pending.submitted_at_ms,
                expired_at_ms=now_ms,
                trigger_reason=pending.trigger_reason,
            )
            self._gateway.cancel_review(review_id)
            return SupervisedStepResult(
                outcome=ReviewOutcome.EXPIRED,
                enforcement_output=pending.enforcement_output,
                trigger_reason=pending.trigger_reason,
            )

        decision = self._gateway.poll_decision(review_id)
        if decision is None:
            return None  # Still waiting

        # Unauthenticated decisions are immediately rejected
        if not decision.authenticated:
            del self._pending[review_id]
            return SupervisedStepResult(
                outcome=ReviewOutcome.DENIED,
                enforcement_output=pending.enforcement_output,
                human_decision=decision,
                trigger_reason=pending.trigger_reason,
            )

        action = decision.action

        # ── DENY ──────────────────────────────────────────────────────────
        if action == HumanDecisionAction.DENY:
            del self._pending[review_id]
            record_human_decision(
                self._hitl_audit,
                human_decision=decision,
                enforcement_output=pending.enforcement_output,
            )
            return SupervisedStepResult(
                outcome=ReviewOutcome.DENIED,
                enforcement_output=pending.enforcement_output,
                human_decision=decision,
                trigger_reason=pending.trigger_reason,
                guidance_constraints=decision.guidance,
            )

        # ── APPROVE ───────────────────────────────────────────────────────
        if action == HumanDecisionAction.APPROVE:
            del self._pending[review_id]
            re_result = self._toctou_reenforce(pending.enforcement_output)
            record_human_decision(
                self._hitl_audit,
                human_decision=decision,
                enforcement_output=pending.enforcement_output,
                re_enforcement_output=re_result,
            )
            if re_result.get("decision") not in ("approve", "project"):
                return SupervisedStepResult(
                    outcome=ReviewOutcome.DENIED,
                    enforcement_output=pending.enforcement_output,
                    human_decision=decision,
                    re_enforcement_output=re_result,
                    trigger_reason=pending.trigger_reason,
                )
            return SupervisedStepResult(
                outcome=ReviewOutcome.EXECUTED,
                enforcement_output=pending.enforcement_output,
                human_decision=decision,
                re_enforcement_output=re_result,
                trigger_reason=pending.trigger_reason,
            )

        # ── MODIFY ────────────────────────────────────────────────────────
        if action == HumanDecisionAction.MODIFY:
            del self._pending[review_id]
            modified = decision.modified_vector
            if modified is None:
                # No replacement vector supplied — treat as deny
                record_human_decision(
                    self._hitl_audit,
                    human_decision=decision,
                    enforcement_output=pending.enforcement_output,
                )
                return SupervisedStepResult(
                    outcome=ReviewOutcome.DENIED,
                    enforcement_output=pending.enforcement_output,
                    human_decision=decision,
                    trigger_reason=pending.trigger_reason,
                )
            # Accept dict directly; for ndarray, zip against original field names
            if hasattr(modified, "items"):
                proposed_values: Dict[str, float] = dict(modified)
            else:
                original = dict(
                    pending.enforcement_output.numerail_result.get("enforced_values") or {}
                )
                field_names = list(original.keys())
                arr = list(modified)
                proposed_values = {
                    k: float(arr[i]) if i < len(arr) else original[k]
                    for i, k in enumerate(field_names)
                }
            re_result = self._toctou_reenforce(pending.enforcement_output, proposed_values)
            record_human_decision(
                self._hitl_audit,
                human_decision=decision,
                enforcement_output=pending.enforcement_output,
                re_enforcement_output=re_result,
            )
            if re_result.get("decision") not in ("approve", "project"):
                return SupervisedStepResult(
                    outcome=ReviewOutcome.DENIED,
                    enforcement_output=pending.enforcement_output,
                    human_decision=decision,
                    re_enforcement_output=re_result,
                    trigger_reason=pending.trigger_reason,
                )
            return SupervisedStepResult(
                outcome=ReviewOutcome.EXECUTED,
                enforcement_output=pending.enforcement_output,
                human_decision=decision,
                re_enforcement_output=re_result,
                trigger_reason=pending.trigger_reason,
            )

        # ── ESCALATE ──────────────────────────────────────────────────────
        if action == HumanDecisionAction.ESCALATE:
            if pending.escalation_depth >= self._triggers.max_escalation_depth:
                del self._pending[review_id]
                record_human_decision(
                    self._hitl_audit,
                    human_decision=decision,
                    enforcement_output=pending.enforcement_output,
                )
                return SupervisedStepResult(
                    outcome=ReviewOutcome.DENIED,
                    enforcement_output=pending.enforcement_output,
                    human_decision=decision,
                    trigger_reason=pending.trigger_reason,
                )
            new_depth = pending.escalation_depth + 1
            new_context: Dict[str, Any] = {**pending.context, "escalation_depth": new_depth}
            new_review_id = self._gateway.submit_for_review(
                pending.enforcement_output,
                pending.trigger_reason,
                new_context,
            )
            del self._pending[review_id]
            self._pending[new_review_id] = PendingAction(
                review_id=new_review_id,
                enforcement_output=pending.enforcement_output,
                submitted_at_ms=self._get_current_time_ms(),
                trigger_reason=pending.trigger_reason,
                context=new_context,
                review_mode=pending.review_mode,
                escalation_depth=new_depth,
                deferred=False,
            )
            return None  # New review pending — caller must poll the new review_id

        # ── DEFER ─────────────────────────────────────────────────────────
        if action == HumanDecisionAction.DEFER:
            if pending.deferred:
                # Second deferral is not permitted — fail closed
                del self._pending[review_id]
                record_human_decision(
                    self._hitl_audit,
                    human_decision=decision,
                    enforcement_output=pending.enforcement_output,
                )
                return SupervisedStepResult(
                    outcome=ReviewOutcome.DENIED,
                    enforcement_output=pending.enforcement_output,
                    human_decision=decision,
                    trigger_reason=pending.trigger_reason,
                )
            # Mark deferred — deadline will extend by one full timeout on next poll
            self._pending[review_id] = PendingAction(
                review_id=review_id,
                enforcement_output=pending.enforcement_output,
                submitted_at_ms=pending.submitted_at_ms,
                trigger_reason=pending.trigger_reason,
                context=pending.context,
                review_mode=pending.review_mode,
                escalation_depth=pending.escalation_depth,
                deferred=True,
            )
            return None  # Reviewer now has an extended window

        # Unknown action — deny defensively
        del self._pending[review_id]
        return SupervisedStepResult(
            outcome=ReviewOutcome.DENIED,
            enforcement_output=pending.enforcement_output,
            human_decision=decision,
            trigger_reason=pending.trigger_reason,
        )

    def resolve_all_pending(self) -> List[Tuple[str, Optional[SupervisedStepResult]]]:
        """Poll all currently pending reviews in one call.

        Returns a list of ``(review_id, result)`` pairs.  Where a review is
        still waiting for a decision, ``result`` is ``None``.

        Note: ESCALATE creates a new review_id.  The new review is NOT included
        in this batch result; it will appear in the next ``resolve_all_pending``
        or ``get_pending_ids`` call.
        """
        review_ids = list(self._pending.keys())
        results: List[Tuple[str, Optional[SupervisedStepResult]]] = []
        for rid in review_ids:
            result = self.resolve_pending(rid)
            results.append((rid, result))
        return results

    def get_pending_count(self) -> int:
        """Return the number of reviews currently awaiting a decision."""
        return len(self._pending)

    def get_pending_ids(self) -> List[str]:
        """Return the review_ids of all currently pending reviews."""
        return list(self._pending.keys())

    @property
    def hitl_audit_records(self) -> List[Dict[str, Any]]:
        """All HITL audit records (human decisions and expiry events)."""
        return self._hitl_audit.records
