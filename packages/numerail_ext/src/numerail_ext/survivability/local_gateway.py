"""Local (in-process) approval gateway for testing and development.

``LocalApprovalGateway`` implements the ``ApprovalGateway`` protocol using
in-memory storage.  It is not suitable for production use — there is no
network, authentication, or persistence — but it provides full coverage of
the HITL protocol for local testing and CI.

Typical usage in tests::

    gateway = LocalApprovalGateway()
    gateway.set_auto_approve()          # all reviews auto-approved

    # or program specific decisions:
    review_id = gateway.submit_for_review(output, "on_reject", {})
    gateway.program_decision(review_id, HumanDecision(
        review_id=review_id,
        action=HumanDecisionAction.APPROVE,
        reviewer_id="test-reviewer",
        authenticated=True,
        timestamp_ms=float(time.time_ns() // 1_000_000),
        reason="looks fine",
    ))
    decision = gateway.poll_decision(review_id)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional

from numerail.protocols import (
    ApprovalGateway,
    HumanDecision,
    HumanDecisionAction,
)


class LocalApprovalGateway:
    """In-process approval gateway for testing and local development.

    Implements the ``ApprovalGateway`` protocol with three modes of
    operation:

    - **Manual** (default): reviews require an explicit ``program_decision``
      call before ``poll_decision`` returns a result.
    - **Auto-approve**: every ``submit_for_review`` immediately creates an
      APPROVE decision (``set_auto_approve``).
    - **Auto-deny**: every ``submit_for_review`` immediately creates a DENY
      decision (``set_auto_deny``).

    Not suitable for production use.  Does not authenticate reviewers
    (``authenticated=True`` on auto-decisions; caller is responsible for
    setting ``authenticated`` correctly on programmed decisions).
    """

    def __init__(self) -> None:
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._decisions: Dict[str, HumanDecision] = {}
        self._review_counter: int = 0
        self._auto_mode: Optional[Literal["approve_all", "deny_all"]] = None
        self._auto_reviewer_id: str = "auto"

    # ── ApprovalGateway protocol ─────────────────────────────────────────

    def submit_for_review(
        self,
        enforcement_output: Any,
        trigger_reason: str,
        context: Dict[str, Any],
    ) -> str:
        """Submit an enforcement result for human review.

        Prints a formatted notification to stdout and, if an auto-mode is
        active, immediately creates the corresponding decision.

        Returns
        -------
        str
            A unique review_id of the form ``"review-N"``.
        """
        review_id = f"review-{self._review_counter}"
        self._review_counter += 1

        self._pending[review_id] = {
            "enforcement_output": enforcement_output,
            "trigger_reason": trigger_reason,
            "context": context,
        }

        # Notification
        result_str = getattr(enforcement_output, "result", enforcement_output)
        print(f"[HITL] Review required: {trigger_reason}")
        print(f"[HITL] Review ID: {review_id}")
        print(f"[HITL] Enforcement result: {result_str}")

        # Auto-mode: immediately create the decision
        if self._auto_mode == "approve_all":
            self._decisions[review_id] = HumanDecision(
                review_id=review_id,
                action=HumanDecisionAction.APPROVE,
                reviewer_id=self._auto_reviewer_id,
                authenticated=True,
                timestamp_ms=float(time.time_ns() // 1_000_000),
                reason="auto-approved by LocalApprovalGateway",
            )
        elif self._auto_mode == "deny_all":
            self._decisions[review_id] = HumanDecision(
                review_id=review_id,
                action=HumanDecisionAction.DENY,
                reviewer_id=self._auto_reviewer_id,
                authenticated=True,
                timestamp_ms=float(time.time_ns() // 1_000_000),
                reason="auto-denied by LocalApprovalGateway",
            )

        return review_id

    def poll_decision(self, review_id: str) -> Optional[HumanDecision]:
        """Non-blocking poll for a human decision.

        If a decision is available, removes it from the store (consumed once)
        and returns it.  Returns ``None`` if no decision has been made yet.
        """
        decision = self._decisions.get(review_id)
        if decision is not None:
            del self._decisions[review_id]
            self._pending.pop(review_id, None)
            return decision
        return None

    def cancel_review(self, review_id: str) -> None:
        """Cancel a pending review, removing it from all internal stores."""
        self._pending.pop(review_id, None)
        self._decisions.pop(review_id, None)

    # ── Testing helpers (not part of ApprovalGateway protocol) ───────────

    def program_decision(self, review_id: str, decision: HumanDecision) -> None:
        """Pre-program a decision to be returned by the next ``poll_decision``.

        Testing helper — not part of the ``ApprovalGateway`` protocol.
        The decision is returned (and consumed) on the next ``poll_decision``
        call for the given ``review_id``.
        """
        self._decisions[review_id] = decision

    def set_auto_approve(self, reviewer_id: str = "auto-approver") -> None:
        """Switch to auto-approve mode.

        Every subsequent ``submit_for_review`` immediately creates an
        APPROVE decision with ``authenticated=True``.
        """
        self._auto_mode = "approve_all"
        self._auto_reviewer_id = reviewer_id

    def set_auto_deny(self, reviewer_id: str = "auto-denier") -> None:
        """Switch to auto-deny mode.

        Every subsequent ``submit_for_review`` immediately creates a
        DENY decision with ``authenticated=True``.
        """
        self._auto_mode = "deny_all"
        self._auto_reviewer_id = reviewer_id

    def set_manual(self) -> None:
        """Switch to manual mode (the default).

        Reviews require an explicit ``program_decision`` call before
        ``poll_decision`` returns a result.
        """
        self._auto_mode = None

    def get_pending_count(self) -> int:
        """Return the number of reviews currently awaiting a decision."""
        return len(self._pending)
