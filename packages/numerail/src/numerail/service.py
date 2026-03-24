"""Numerail production runtime service.

Wraps the unchanged enforcement engine in transactional infrastructure:
scoped authorization, Protocol-based repositories, strict policy parsing,
audit trail, metrics collection, and outbox publishing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from numerail.engine import NumerailSystem, RollbackResult
from numerail.errors import AuthorizationError
from numerail.protocols import ServiceRequest
from numerail.parser import PolicyParser


class NumerailRuntimeService:
    """Production orchestration layer around the unchanged engine.

    Transactional flow: authorize → load policy → lock budgets → strict parse →
    build engine → enforce → persist delta/ledger/audit/metrics/outbox → commit.
    """

    def __init__(self, *, transactions, authz, policy_repo, runtime_repo,
                 ledger_repo, audit_repo, metrics_repo, outbox_repo,
                 parser=None):
        self._txn = transactions
        self._authz = authz
        self._policy_repo = policy_repo
        self._runtime_repo = runtime_repo
        self._ledger_repo = ledger_repo
        self._audit_repo = audit_repo
        self._metrics_repo = metrics_repo
        self._outbox_repo = outbox_repo
        self._parser = parser or PolicyParser()

    def enforce(self, *, policy_id, proposed_action, action_id, request):
        self._authz.require(request.scopes, "enforce")
        if request.trusted_context:
            self._authz.require_any(request.scopes, {"trusted:inject"})

        with self._txn():
            config_dict = self._policy_repo.load_active(policy_id)
            validated = self._parser.parse(config_dict)
            head = self._runtime_repo.lock_runtime_head(policy_id)
            budget_remaining = dict(self._runtime_repo.lock_budget_rows(policy_id))

            system = NumerailSystem.from_config(validated)
            init_status = system.budget_status()
            for bname, remaining in budget_remaining.items():
                if bname in init_status:
                    system.sync_budget_consumed(bname, init_status[bname]["initial"] - remaining)

            tf = validated.get("trusted_fields", [])
            if tf:
                system.set_trusted_fields(frozenset(tf))

            raw_values = dict(proposed_action)

            # Snapshot budget consumed totals BEFORE enforcement
            pre_consumed = {}
            if system.has_budgets:
                pre_status = system.budget_status()
                for bname in pre_status:
                    pre_consumed[bname] = pre_status[bname]["consumed"]

            result = system.enforce(proposed_action, action_id=action_id,
                                    trusted_context=request.trusted_context)

            decision = result.output.result.value
            enforced_values = result.enforced_values if decision != "reject" else None
            merged_values = result.feedback.get("merged_values", raw_values)

            # Compute per-action delta by diffing consumed totals
            budget_delta = {}
            if system.has_budgets:
                post_status = system.budget_status()
                for bname, pre in pre_consumed.items():
                    post = post_status[bname]["consumed"]
                    delta = post - pre
                    if abs(delta) > 1e-15:
                        budget_delta[bname] = delta

            if decision in ("approve", "project") and budget_delta:
                self._runtime_repo.apply_budget_delta(policy_id, budget_delta)

            audit_record = {
                "action_id": action_id, "decision": decision,
                "region_version": result.output.region_version,
                "timestamp": result.output.timestamp,
                "distance": result.output.distance,
                "violated": list(result.output.violated_constraints),
                "binding": list(result.output.binding_constraints),
                "solver": result.output.solver_method,
            }
            audit_hash = self._audit_repo.append_decision(action_id=action_id, record=audit_record)

            self._ledger_repo.insert_decision(
                policy_id=policy_id, action_id=action_id,
                raw_values=raw_values, merged_values=merged_values,
                enforced_values=enforced_values, decision=decision,
                budget_delta=budget_delta, region_version=result.output.region_version,
                timestamp=result.output.timestamp, scopes=list(request.scopes),
                audit_hash=audit_hash,
            )

            self._metrics_repo.record_enforcement(policy_id=policy_id, result=result)

            if (decision in ("approve", "project") and request.execution_topic
                    and enforced_values is not None):
                self._outbox_repo.enqueue(
                    topic=request.execution_topic, action_id=action_id,
                    payload=enforced_values, audit_hash=audit_hash,
                )

            return {"enforced_values": enforced_values, "decision": decision,
                    "feedback": result.feedback, "audit_hash": audit_hash,
                    "action_id": action_id}

    def rollback(self, *, action_id, scopes):
        self._authz.require_any(scopes, {"rollback"})
        with self._txn():
            if self._ledger_repo.is_rolled_back(action_id):
                raise ValueError(f"action_id={action_id!r} already rolled back")
            delta = dict(self._ledger_repo.read_budget_delta(action_id))
            self._runtime_repo.restore_budget_delta(delta)
            self._ledger_repo.mark_rolled_back(action_id)
            audit_hash = self._audit_repo.append_rollback(action_id=action_id, delta=delta)
            self._metrics_repo.record_rollback(action_id=action_id)
            return RollbackResult(rolled_back=True, audit_hash=audit_hash)
