"""Receipt-against-grant validation for the post-execution trust boundary.

This is the supervisory analogue of V5's post-check.  V5 guarantees that
the *emitted* vector is safe.  This module guarantees that the *executed*
action matches the emitted grant.
"""

from __future__ import annotations

from .types import ExecutableGrant, ExecutionReceipt


class ReceiptValidationError(Exception):
    """Raised when an execution receipt does not match its grant."""


def validate_receipt_against_grant(
    *, grant: ExecutableGrant, receipt: ExecutionReceipt,
) -> None:
    """Verify that an execution receipt is consistent with its grant.

    Checks:
        1. action_id identity match
        2. state_version consistency
        3. payload_digest tamper detection
        4. execution within freshness window

    Raises :class:`ReceiptValidationError` on any mismatch.
    """
    if receipt.action_id != grant.action_id:
        raise ReceiptValidationError(
            f"action_id mismatch: grant={grant.action_id!r}, "
            f"receipt={receipt.action_id!r}"
        )
    if receipt.state_version != grant.state_version:
        raise ReceiptValidationError(
            f"state_version mismatch: grant={grant.state_version}, "
            f"receipt={receipt.state_version}"
        )
    if receipt.payload_digest != grant.payload_digest:
        raise ReceiptValidationError(
            f"payload_digest mismatch: grant digest={grant.payload_digest[:16]}..., "
            f"receipt digest={receipt.payload_digest[:16]}..."
        )
    if receipt.observed_at_ns > grant.expires_at_ns:
        raise ReceiptValidationError(
            f"receipt observed_at_ns ({receipt.observed_at_ns}) exceeds "
            f"grant expires_at_ns ({grant.expires_at_ns})"
        )
