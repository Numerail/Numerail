"""Numerail production Protocol interfaces.

These define the repository contracts that production deployments must implement.
The engine itself is repository-agnostic; these Protocols are the integration surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

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
