"""Numerail production-layer exceptions."""

from numerail.engine import NumerailError


class AuthorizationError(NumerailError):
    """Raised when a caller lacks the required scope."""
