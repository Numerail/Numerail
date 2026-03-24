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
from typing import Dict

from numerail.engine import RollbackResult, _utc_now
from numerail.errors import AuthorizationError
from numerail.protocols import LockedRuntimeHead, ServiceRequest
from numerail.parser import PolicyParser
from numerail.service import NumerailRuntimeService


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
    """

    def __init__(self, config_dict):
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

    def enforce(self, proposed_action, *, action_id=None, trusted_context=None,
                execution_topic=None):
        if action_id is None:
            self._counter += 1; action_id = f"local_{self._counter}"
        scopes = ["enforce"] + (["trusted:inject"] if trusted_context else [])
        return self._service.enforce(
            policy_id=self._policy_id, proposed_action=proposed_action,
            action_id=action_id,
            request=ServiceRequest(scopes=scopes, trusted_context=trusted_context,
                                   execution_topic=execution_topic),
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
