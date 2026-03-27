"""Numerail production Protocol interfaces.

These define the repository contracts that production deployments must implement.
The engine itself is repository-agnostic; these Protocols are the integration surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    pass  # Forward-reference guard for EnforcementOutput


__all__ = [
    # Existing protocols and types
    "TransactionManager",
    "AuthorizationService",
    "PolicyRepository",
    "RuntimeRepository",
    "LedgerRepository",
    "AuditRepository",
    "MetricsRepository",
    "OutboxRepository",
    "LockedRuntimeHead",
    "ServiceRequest",
    "TrustedContextProvider",
    # HITL types
    "HumanDecisionAction",
    "HumanDecision",
    "ReviewOutcome",
    "ApprovalGateway",
]


class TransactionManager(Protocol):
    def __enter__(self) -> "TransactionManager": ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


class AuthorizationService(Protocol):
    def require(self, scopes: Sequence[str], required: str) -> None: ...
    def require_any(self, scopes: Sequence[str], allowed: set) -> None: ...


class PolicyRepository(Protocol):
    def load_active(self, policy_id: str) -> dict: ...


class RuntimeRepository(Protocol):
    def lock_runtime_head(self, policy_id: str) -> "LockedRuntimeHead": ...
    def lock_budget_rows(self, policy_id: str) -> Dict[str, float]: ...
    def apply_budget_delta(self, policy_id: str, delta: Dict[str, float]) -> Dict[str, float]: ...
    def restore_budget_delta(self, delta: Dict[str, float]) -> Dict[str, float]: ...


class LedgerRepository(Protocol):
    def insert_decision(self, **kwargs) -> None: ...
    def read_budget_delta(self, action_id: str) -> Dict[str, float]: ...
    def mark_rolled_back(self, action_id: str) -> None: ...
    def is_rolled_back(self, action_id: str) -> bool: ...


class AuditRepository(Protocol):
    def append_decision(self, *, action_id: str, record: dict) -> str: ...
    def append_rollback(self, *, action_id: str, delta: Dict[str, float]) -> str: ...


class MetricsRepository(Protocol):
    def record_enforcement(self, *, policy_id: str, result: Any) -> None: ...
    def record_rollback(self, *, action_id: str) -> None: ...


class OutboxRepository(Protocol):
    def enqueue(self, *, topic: str, action_id: str, payload: Dict[str, float], audit_hash: str) -> None: ...


@dataclass(frozen=True)
class LockedRuntimeHead:
    policy_version: str
    audit_prev_hash: Optional[str]
    timestamp: str


@dataclass(frozen=True)
class ServiceRequest:
    scopes: Sequence[str] = field(default_factory=lambda: ["enforce"])
    trusted_context: Optional[Dict[str, float]] = None
    execution_topic: Optional[str] = None


class TrustedContextProvider(Protocol):
    """Provider of server-authoritative trusted field values.

    Trusted fields are values that the AI must not control. The provider's
    values overwrite the AI's proposed values for matching schema fields
    before enforcement, ensuring that safety-critical information always
    comes from server-authoritative sources rather than the model's claims.

    ``current_time_ms`` is the most critical trusted field: every temporal
    constraint — shift windows, freshness checks, budget period boundaries,
    expiry windows — depends on the wall-clock time.  Allowing the model to
    supply its own time value would let it manipulate all time-based
    constraints simultaneously with a single field.

    Implementing a provider
    -----------------------
    Read each value from your monitoring stack or NTP at call time.  Do not
    cache values across calls; the provider is called once per enforce() and
    must return a fresh reading.

    Example::

        class MonitoringStackProvider:
            def get_trusted_context(self) -> dict[str, float]:
                return {
                    "current_time_ms": float(ntp_client.time_ms()),
                    "current_gpu_util": metrics.gpu_utilization(),
                    "current_api_util": metrics.api_utilization(),
                }

            @property
            def trusted_field_names(self) -> frozenset[str]:
                return frozenset({
                    "current_time_ms", "current_gpu_util", "current_api_util",
                })
    """

    def get_trusted_context(self) -> Dict[str, float]:
        """Return a fresh dict of field_name → authoritative_value.

        Called once per enforce() call.  The returned values overwrite the
        AI's proposed values for any key that exists in the active schema.
        Keys that are not in the schema are silently skipped.
        """
        ...

    @property
    def trusted_field_names(self) -> FrozenSet[str]:
        """The set of field names this provider considers authoritative.

        Used for documentation and audit purposes; the runtime injection
        iterates ``get_trusted_context()`` directly.
        """
        ...


# ── Human-in-the-loop types ──────────────────────────────────────────────


class HumanDecisionAction(str, Enum):
    """Actions a human reviewer can take on a pending enforcement decision.

    - APPROVE: reviewer authorizes execution of the enforced action as-is
    - DENY: reviewer blocks execution; may include guidance bounds for the
      next attempt via the ``guidance`` field of ``HumanDecision``
    - MODIFY: reviewer proposes a different vector, which will be re-enforced
      before execution; the modified vector is provided in
      ``HumanDecision.modified_vector``
    - ESCALATE: reviewer lacks authority or context; action is forwarded to
      a higher-authority reviewer (up to ``max_escalation_depth`` times)
    - DEFER: reviewer needs more time; the review timeout is extended once
    """

    APPROVE = "approve"
    DENY = "deny"
    MODIFY = "modify"
    ESCALATE = "escalate"
    DEFER = "defer"


@dataclass(frozen=True)
class HumanDecision:
    """A human reviewer's decision on a pending enforcement action.

    The ``authenticated`` field must be ``True`` for the decision to be
    accepted — the ``ApprovalGateway`` implementation is responsible for
    verifying reviewer identity.  The ``SupervisedGovernor`` will reject
    decisions where ``authenticated`` is ``False``.

    Attributes
    ----------
    review_id : str
        The review identifier returned by ``ApprovalGateway.submit_for_review``.
    action : HumanDecisionAction
        The reviewer's chosen action.
    reviewer_id : str
        Identifier of the authenticated reviewer.
    authenticated : bool
        Whether the reviewer's identity has been verified by the gateway.
    timestamp_ms : float
        Wall-clock time of the decision in milliseconds since the Unix epoch.
    reason : str
        Human-readable explanation of the decision.
    modified_vector : np.ndarray or None
        Only present when ``action`` is ``MODIFY``.  The reviewer's proposed
        replacement vector, which will be re-enforced before execution.
    guidance : dict or None
        Optional bounds hints when ``action`` is ``DENY``, mapping field name
        to ``(min_value, max_value)``.  Provided as advisory guidance for the
        next attempt; the engine is the enforced authority.
    escalation_depth : int
        Number of times this review has been escalated (0 = not escalated).
    """

    review_id: str
    action: HumanDecisionAction
    reviewer_id: str
    authenticated: bool
    timestamp_ms: float
    reason: str
    modified_vector: Optional[np.ndarray] = None
    guidance: Optional[Dict[str, Tuple[float, float]]] = None
    escalation_depth: int = 0


class ReviewOutcome(str, Enum):
    """Final outcome of a supervised enforcement cycle.

    - EXECUTED: action passed enforcement (and review if triggered) and was
      committed to execution
    - PENDING_REVIEW: action passed enforcement but a review trigger fired;
      awaiting a human decision
    - DENIED: action was rejected by enforcement, denied by a human reviewer,
      or the reviewer's modification was infeasible after re-enforcement
    - EXPIRED: review timeout reached with no human decision; action denied
      by default (fail-closed)
    """

    EXECUTED = "executed"
    PENDING_REVIEW = "pending_review"
    DENIED = "denied"
    EXPIRED = "expired"


class ApprovalGateway(Protocol):
    """Abstract gateway for human-in-the-loop approval.

    Implementations MUST authenticate the reviewer before accepting a
    decision.  The ``SupervisedGovernor`` will reject decisions where
    ``authenticated`` is ``False``.

    The notification mechanism (Slack, dashboard, email, mobile, terminal)
    is deployment-specific and outside the scope of this protocol.
    Implementations are free to notify reviewers synchronously or
    asynchronously on ``submit_for_review``.
    """

    def submit_for_review(
        self,
        enforcement_output: Any,
        trigger_reason: str,
        context: Dict[str, Any],
    ) -> str:
        """Submit an action for human review. Returns a review_id.

        Parameters
        ----------
        enforcement_output : EnforcementOutput
            The kernel enforcement result to be reviewed.
        trigger_reason : str
            Human-readable description of why the review was triggered
            (e.g. ``"on_reject"`` or ``"on_safe_stop_entry"``).
        context : dict
            Operational context for the reviewer.  Should include
            ``breaker_mode``, ``budget_state``, ``policy_version``,
            and any relevant operational context.

        Returns
        -------
        str
            A unique review_id that can be passed to ``poll_decision``
            and ``cancel_review``.
        """
        ...

    def poll_decision(self, review_id: str) -> Optional[HumanDecision]:
        """Non-blocking poll for a human decision.

        Returns ``None`` if the reviewer has not decided yet.  Callers
        should poll periodically and handle the ``None`` case by waiting
        or timing out according to ``HumanReviewTriggers.review_timeout_seconds``.
        """
        ...

    def cancel_review(self, review_id: str) -> None:
        """Cancel a pending review.

        Used when the action expires or is superseded.  Implementations
        should notify the reviewer that the review is no longer needed.
        """
        ...
