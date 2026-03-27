"""Numerail local-mode system.

Wires in-memory implementations of every Protocol into the real
NumerailRuntimeService. Same scope checks, same ledger writes,
same audit records, same transactional flow — state lives in memory.

Usage:
    local = NumerailSystemLocal(config_dict)
    result = local.enforce({"amount": 500.0, "rate": 0.05})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time as _time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from numerail.engine import RollbackResult, _utc_now
from numerail.errors import AuthorizationError
from numerail.protocols import LockedRuntimeHead, ServiceRequest
from numerail.parser import PolicyParser
from numerail.service import NumerailRuntimeService

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Distinguishes "omitted" from explicit None


# ── Trusted context types ────────────────────────────────────────────────


class DefaultTimeProvider:
    """Default TrustedContextProvider: injects the current wall-clock time.

    Returns ``current_time_ms`` — milliseconds since the Unix epoch — as the
    single trusted field.

    **Why milliseconds, not nanoseconds?**  float64 can represent integers
    exactly up to 2^53 ≈ 9.0 × 10^15.  The current Unix time in milliseconds
    is ~1.7 × 10^12, leaving five orders of magnitude of headroom.  Unix time
    in nanoseconds is ~1.7 × 10^18, which exceeds the float64 exact-integer
    range, causing silent rounding errors in temporal constraints.
    """

    def get_trusted_context(self) -> Dict[str, float]:
        """Return the current wall-clock time in milliseconds."""
        return {"current_time_ms": float(_time.time_ns() // 1_000_000)}

    @property
    def trusted_field_names(self) -> frozenset:
        return frozenset({"current_time_ms"})


@dataclass
class SystemEnforcementResult:
    """Enforcement result with trusted context injection metadata.

    Wraps the kernel service output together with the original AI proposal
    and any field overrides applied by the TrustedContextProvider before
    enforcement.

    Attributes
    ----------
    kernel_output : dict
        Full service response: ``decision``, ``enforced_values``,
        ``feedback``, ``audit_hash``.
    original_proposal : dict
        The values the AI proposed, before trusted injection.
    trusted_overrides : dict
        ``{field_name: (ai_proposed_value, authoritative_value)}`` for every
        field where the provider's value differed from the AI's claim by more
        than 1e-9.  Empty when no overwrite occurred.

    Notes
    -----
    Supports dict-style key access (``result["decision"]``, ``result.get(...)``,
    ``"key" in result``) for backward compatibility with code that previously
    consumed the raw kernel_output dict directly.
    """

    kernel_output: dict
    original_proposal: dict
    trusted_overrides: Dict[str, Tuple[float, float]]

    # ── Backward-compatible dict-style access ─────────────────────────

    def __getitem__(self, key: str):
        return self.kernel_output[key]

    def __contains__(self, key: str) -> bool:
        return key in self.kernel_output

    def get(self, key: str, default=None):
        return self.kernel_output.get(key, default)


# ── In-memory Protocol implementations ──────────────────────────────────


class _InMemoryTxn:
    """No-op transaction manager for local/test mode.

    Does not simulate atomicity. Production deployments must provide a
    real TransactionManager implementation with ACID semantics."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _InMemoryAuthz:
    def require(self, scopes, required):
        if required not in set(scopes):
            raise AuthorizationError(f"Missing scope: {required}")
    def require_any(self, scopes, allowed):
        if not (set(scopes) & set(allowed)):
            raise AuthorizationError(f"Missing one of: {allowed}")


class _InMemoryPolicyRepo:
    def __init__(self, config): self._config = dict(config)
    def load_active(self, policy_id): return dict(self._config)


class _InMemoryRuntimeRepo:
    def __init__(self, initial):
        self._remaining = dict(initial)
    def lock_runtime_head(self, policy_id):
        return LockedRuntimeHead("local", None, _utc_now())
    def lock_budget_rows(self, policy_id):
        return dict(self._remaining)
    def apply_budget_delta(self, policy_id, delta):
        for k, v in delta.items():
            self._remaining[k] = self._remaining.get(k, 0.0) - float(v)
        return dict(self._remaining)
    def restore_budget_delta(self, delta):
        for k, v in delta.items():
            self._remaining[k] = self._remaining.get(k, 0.0) + float(v)
        return dict(self._remaining)


class _InMemoryLedgerRepo:
    def __init__(self): self._rows = {}
    def insert_decision(self, **kw):
        self._rows[kw["action_id"]] = {**kw, "rolled_back": False}
    def read_budget_delta(self, action_id):
        return dict(self._rows[action_id].get("budget_delta", {}))
    def mark_rolled_back(self, action_id):
        self._rows[action_id]["rolled_back"] = True
    def is_rolled_back(self, action_id):
        return bool(self._rows.get(action_id, {}).get("rolled_back", False))


class _InMemoryAuditRepo:
    def __init__(self): self._last = None; self.records = []
    def append_decision(self, *, action_id, record):
        payload = {**record, "prev_hash": self._last}
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        self._last = h; self.records.append({"type": "decision", "hash": h, **payload})
        return h
    def append_rollback(self, *, action_id, delta):
        payload = {"action_id": action_id, "delta": dict(delta), "prev_hash": self._last}
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self._last = h; self.records.append({"type": "rollback", "hash": h, **payload})
        return h


class _InMemoryMetricsRepo:
    def __init__(self): self.enforcements = []; self.rollbacks = []
    def record_enforcement(self, *, policy_id, result):
        self.enforcements.append((policy_id, result.output.result.value))
    def record_rollback(self, *, action_id):
        self.rollbacks.append(action_id)


class _InMemoryOutboxRepo:
    def __init__(self): self.events = []
    def enqueue(self, *, topic, action_id, payload, audit_hash):
        self.events.append({"topic": topic, "action_id": action_id,
                           "payload": dict(payload), "audit_hash": audit_hash})


# ── Local system facade ─────────────────────────────────────────────────


class NumerailSystemLocal:
    """Local-mode system exercising the full production code path.

    Wires in-memory implementations of every Protocol into the real
    NumerailRuntimeService. Same scope checks, same ledger writes,
    same audit records, same transactional flow — state lives in memory.

    Parameters
    ----------
    config_dict : dict
        V5-compatible policy configuration.
    trusted_context_provider : TrustedContextProvider, optional
        Provider of server-authoritative field values.  Its values overwrite
        the AI's proposed values for matching schema fields before each
        enforcement call.  Defaults to ``DefaultTimeProvider``, which injects
        the current wall-clock time as ``current_time_ms``.

        Pass ``trusted_context_provider=None`` explicitly only if you want
        to disable automatic injection entirely for a specific instance.
    """

    def __init__(self, config_dict, trusted_context_provider=_SENTINEL):
        self._config = dict(config_dict)
        self._policy_id = config_dict.get("policy_id", "local")
        initial_budgets = {b["name"]: float(b.get("initial", 0))
                          for b in config_dict.get("budgets", [])}
        self._runtime_repo = _InMemoryRuntimeRepo(initial_budgets)
        self._ledger_repo = _InMemoryLedgerRepo()
        self._audit_repo = _InMemoryAuditRepo()
        self._metrics_repo = _InMemoryMetricsRepo()
        self._outbox_repo = _InMemoryOutboxRepo()
        self._service = NumerailRuntimeService(
            transactions=_InMemoryTxn, authz=_InMemoryAuthz(),
            policy_repo=_InMemoryPolicyRepo(config_dict),
            runtime_repo=self._runtime_repo, ledger_repo=self._ledger_repo,
            audit_repo=self._audit_repo, metrics_repo=self._metrics_repo,
            outbox_repo=self._outbox_repo, parser=PolicyParser(),
        )
        self._counter = 0
        # Default to DefaultTimeProvider if caller omitted the argument entirely.
        # Passing None explicitly disables injection.
        if trusted_context_provider is _SENTINEL:
            trusted_context_provider = DefaultTimeProvider()
        self._trusted_provider: Optional[object] = trusted_context_provider
        # Build a fast lookup of schema field names for O(1) membership tests.
        raw_fields = config_dict.get("schema", {}).get("fields", [])
        # Schema fields may be plain strings or dicts with a "name" key.
        self._schema_field_names: frozenset = frozenset(
            f["name"] if isinstance(f, dict) else f for f in raw_fields
        )

    def enforce(self, proposed_action, *, action_id=None, trusted_context=None,
                execution_topic=None):
        """Enforce the proposed action, applying trusted context injection first.

        Returns
        -------
        SystemEnforcementResult
            Wraps the kernel output with the original proposal and any
            provider-injected overrides.  Supports dict-style access
            (``result["decision"]``) for backward compatibility.
        """
        if action_id is None:
            self._counter += 1
            action_id = f"local_{self._counter}"

        # ── Trusted context injection ────────────────────────────────
        original_proposal: dict = dict(proposed_action)
        merged: dict = dict(proposed_action)
        trusted_overrides: Dict[str, Tuple[float, float]] = {}

        if self._trusted_provider is not None:
            tc = self._trusted_provider.get_trusted_context()
            for field_name, authoritative in tc.items():
                authoritative = float(authoritative)
                if field_name not in self._schema_field_names:
                    logger.debug(
                        "TrustedContextProvider: field %r not in schema, skipping",
                        field_name,
                    )
                    continue
                ai_value = float(merged.get(field_name, authoritative))
                if abs(ai_value - authoritative) > 1e-9:
                    logger.warning(
                        "TrustedContextProvider: field %r overwritten — "
                        "AI proposed %.6g, authoritative %.6g",
                        field_name,
                        ai_value,
                        authoritative,
                    )
                    trusted_overrides[field_name] = (ai_value, authoritative)
                merged[field_name] = authoritative

        # ── Kernel enforcement ───────────────────────────────────────
        scopes = ["enforce"] + (["trusted:inject"] if trusted_context else [])
        kernel_output = self._service.enforce(
            policy_id=self._policy_id,
            proposed_action=merged,
            action_id=action_id,
            request=ServiceRequest(
                scopes=scopes,
                trusted_context=trusted_context,
                execution_topic=execution_topic,
            ),
        )

        return SystemEnforcementResult(
            kernel_output=kernel_output,
            original_proposal=original_proposal,
            trusted_overrides=trusted_overrides,
        )

    def rollback(self, action_id):
        return self._service.rollback(action_id=action_id, scopes=["rollback"])

    @property
    def budget_remaining(self): return dict(self._runtime_repo._remaining)
    @property
    def ledger(self): return dict(self._ledger_repo._rows)
    @property
    def audit_records(self): return list(self._audit_repo.records)
    @property
    def metrics(self):
        return {"enforcements": list(self._metrics_repo.enforcements),
                "rollbacks": list(self._metrics_repo.rollbacks)}
    @property
    def outbox_events(self): return list(self._outbox_repo.events)
