"""Numerail production Protocol interfaces.

These define the repository contracts that production deployments must implement.
The engine itself is repository-agnostic; these Protocols are the integration surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Sequence

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


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
