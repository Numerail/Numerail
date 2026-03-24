"""Local-mode NumerailBackend adapter wrapping NumerailSystemLocal.

Provides the :class:`NumerailBackend` Protocol interface using in-memory V5
state.  Rebuilds the ``NumerailSystemLocal`` on each ``set_active_config()``
call so that envelope-derived ceilings are always current.

Budget keys returned by ``budget_remaining()`` match the canonical V5
``BudgetSpec.name`` values produced by the policy builder: ``gpu_shift``,
``external_api_shift``, ``mutation_shift``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from numerail.local import NumerailSystemLocal


class LocalNumerailBackend:
    """In-memory V5 backend for development, testing, and single-process use."""

    def __init__(self) -> None:
        self._system: Optional[NumerailSystemLocal] = None

    def budget_remaining(self) -> Mapping[str, float]:
        if self._system is None:
            return {}
        return dict(self._system.budget_remaining)

    def set_active_config(self, config: Mapping[str, Any]) -> None:
        self._system = NumerailSystemLocal(dict(config))

    def enforce(
        self,
        *,
        policy_id: str,
        proposed_action: Mapping[str, float],
        action_id: str,
        trusted_context: Optional[Mapping[str, float]] = None,
        execution_topic: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if self._system is None:
            raise RuntimeError("No active config — call set_active_config() first")
        return self._system.enforce(
            dict(proposed_action),
            action_id=action_id,
            trusted_context=dict(trusted_context) if trusted_context else None,
            execution_topic=execution_topic,
        )

    def rollback(self, *, action_id: str) -> Any:
        if self._system is None:
            raise RuntimeError("No active config — call set_active_config() first")
        return self._system.rollback(action_id)
