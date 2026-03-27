"""Tests for TrustedContextProvider, DefaultTimeProvider, and SystemEnforcementResult.

Covers:
  - DefaultTimeProvider returns a current timestamp within tolerance
  - Trusted field injection overwrites AI-proposed values
  - Original proposal is preserved unchanged
  - trusted_overrides dict is populated correctly
  - Warning is logged on mismatch
  - Fields absent from the schema are skipped (not an error)
  - Custom providers with multiple fields
  - Backward compatibility: NumerailSystemLocal with no provider argument
  - No override recorded when AI and provider values match
  - StateTransitionGovernor applies provider values to the telemetry snapshot
"""

from __future__ import annotations

import logging
import time as _time
from typing import Dict

import pytest

from numerail.local import DefaultTimeProvider, NumerailSystemLocal, SystemEnforcementResult


# ── Minimal test configs ──────────────────────────────────────────────────────

def _minimal_config(*extra_fields: str) -> dict:
    """Policy with 'amount' plus any requested extra fields."""
    fields = ["amount"] + list(extra_fields)
    n = len(fields)
    idx = {f: i for i, f in enumerate(fields)}

    def _row(field, coeff):
        r = [0.0] * n
        r[idx[field]] = coeff
        return r

    # Upper and lower bounds for every field
    A, b, names = [], [], []
    bounds = {
        "amount": 1000.0,
        "current_time_ms": 1e13,
        "gpu_utilization": 1.0,
    }
    for f in fields:
        upper = bounds.get(f, 1e12)
        A.append(_row(f,  1.0)); b.append(upper);  names.append(f"max_{f}")
        A.append(_row(f, -1.0)); b.append(0.0);    names.append(f"min_{f}")

    return {
        "policy_id": "test_trusted_context",
        "schema": {"fields": fields},
        "polytope": {"A": A, "b": b, "names": names},
        "constraints": [],
        "budgets": [],
    }


_NO_PROVIDER = object()  # sentinel: caller omitted the provider argument


def _local(*extra_fields, provider=_NO_PROVIDER) -> NumerailSystemLocal:
    cfg = _minimal_config(*extra_fields)
    if provider is _NO_PROVIDER:
        return NumerailSystemLocal(cfg)
    return NumerailSystemLocal(cfg, trusted_context_provider=provider)


# ── Mock providers ────────────────────────────────────────────────────────────

class _FixedProvider:
    """Returns a fixed dict of trusted values."""

    def __init__(self, values: Dict[str, float]):
        self._values = dict(values)

    def get_trusted_context(self) -> Dict[str, float]:
        return dict(self._values)

    @property
    def trusted_field_names(self) -> frozenset:
        return frozenset(self._values)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDefaultProvider:
    def test_default_provider_returns_current_time_ms(self):
        """Returned value is within 1000 ms of the system clock."""
        provider = DefaultTimeProvider()
        before = _time.time_ns() // 1_000_000
        result = provider.get_trusted_context()
        after = _time.time_ns() // 1_000_000

        assert "current_time_ms" in result
        t = result["current_time_ms"]
        assert isinstance(t, float)
        assert before - 1000 <= t <= after + 1000, (
            f"current_time_ms {t} not within 1000 ms of [{before}, {after}]"
        )

    def test_default_provider_trusted_field_names(self):
        provider = DefaultTimeProvider()
        assert provider.trusted_field_names == frozenset({"current_time_ms"})


class TestTrustedFieldOverwrite:
    def test_trusted_field_overwrite(self):
        """Provider value replaces AI-proposed value in the enforced output."""
        provider = _FixedProvider({"current_time_ms": 1000.0})
        local = _local("current_time_ms", provider=provider)

        result = local.enforce(
            {"amount": 100.0, "current_time_ms": 9999.0},
            action_id="overwrite_test",
        )

        assert isinstance(result, SystemEnforcementResult)
        ev = result["enforced_values"]
        assert abs(ev["current_time_ms"] - 1000.0) < 1e-6, (
            f"Expected enforced current_time_ms ≈ 1000.0, got {ev['current_time_ms']}"
        )

    def test_original_proposal_preserved(self):
        """original_proposal retains the AI's value before injection."""
        provider = _FixedProvider({"current_time_ms": 1000.0})
        local = _local("current_time_ms", provider=provider)

        result = local.enforce(
            {"amount": 100.0, "current_time_ms": 9999.0},
            action_id="orig_test",
        )

        assert abs(result.original_proposal["current_time_ms"] - 9999.0) < 1e-9

    def test_trusted_overrides_populated(self):
        """trusted_overrides records (ai_proposed, authoritative) for each override."""
        provider = _FixedProvider({"current_time_ms": 1000.0})
        local = _local("current_time_ms", provider=provider)

        result = local.enforce(
            {"amount": 100.0, "current_time_ms": 9999.0},
            action_id="overrides_test",
        )

        assert "current_time_ms" in result.trusted_overrides
        ai_val, auth_val = result.trusted_overrides["current_time_ms"]
        assert abs(ai_val - 9999.0) < 1e-9
        assert abs(auth_val - 1000.0) < 1e-9

    def test_warning_logged_on_mismatch(self, caplog):
        """A WARNING is emitted when the AI's value differs from authoritative."""
        provider = _FixedProvider({"current_time_ms": 1000.0})
        local = _local("current_time_ms", provider=provider)

        with caplog.at_level(logging.WARNING, logger="numerail.local"):
            local.enforce(
                {"amount": 100.0, "current_time_ms": 9999.0},
                action_id="warn_test",
            )

        assert any("current_time_ms" in r.message for r in caplog.records), (
            f"Expected warning mentioning 'current_time_ms'; got: {[r.message for r in caplog.records]}"
        )
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_no_overwrite_when_values_match(self):
        """When AI and provider agree, trusted_overrides is empty."""
        provider = _FixedProvider({"current_time_ms": 500.0})
        local = _local("current_time_ms", provider=provider)

        result = local.enforce(
            {"amount": 100.0, "current_time_ms": 500.0},
            action_id="match_test",
        )

        assert result.trusted_overrides == {}, (
            f"Expected empty trusted_overrides when values match; got {result.trusted_overrides}"
        )


class TestFieldNotInSchema:
    def test_field_not_in_schema_skipped(self, caplog):
        """A provider field absent from the schema is skipped without error."""
        provider = _FixedProvider({"unknown_field_xyz": 42.0})
        local = _local(provider=provider)  # schema has only "amount"

        with caplog.at_level(logging.DEBUG, logger="numerail.local"):
            result = local.enforce({"amount": 100.0}, action_id="skip_test")

        assert result["decision"] in ("approve", "project", "reject")
        assert "unknown_field_xyz" not in result.trusted_overrides
        assert any("unknown_field_xyz" in r.message for r in caplog.records), (
            "Expected a DEBUG log for the unknown field"
        )


class TestCustomProvider:
    def test_custom_provider(self):
        """A provider returning multiple fields injects all of them."""
        provider = _FixedProvider({
            "gpu_utilization": 0.75,
            "current_time_ms": 2000.0,
        })
        local = _local("gpu_utilization", "current_time_ms", provider=provider)

        result = local.enforce(
            {"amount": 100.0, "gpu_utilization": 0.10, "current_time_ms": 9999.0},
            action_id="custom_test",
        )

        ev = result["enforced_values"]
        assert abs(ev["gpu_utilization"] - 0.75) < 1e-6
        assert abs(ev["current_time_ms"] - 2000.0) < 1e-6
        assert "gpu_utilization" in result.trusted_overrides
        assert "current_time_ms" in result.trusted_overrides


class TestBackwardCompatibility:
    def test_no_provider_backward_compatible(self):
        """Constructing NumerailSystemLocal with no provider arg still works.

        The default DefaultTimeProvider is used; since 'current_time_ms' is not
        in the test schema, the field is silently skipped and enforcement proceeds
        normally.
        """
        local = NumerailSystemLocal(_minimal_config())  # no provider argument
        result = local.enforce({"amount": 500.0}, action_id="compat_test")
        assert result["decision"] in ("approve", "project", "reject")

    def test_dict_style_access(self):
        """SystemEnforcementResult supports dict-style access for backward compat."""
        provider = _FixedProvider({})
        local = _local(provider=provider)
        result = local.enforce({"amount": 50.0}, action_id="dict_test")

        assert "decision" in result
        assert result["decision"] in ("approve", "project", "reject")
        assert result.get("decision") in ("approve", "project", "reject")
        assert result.get("nonexistent_key", "fallback") == "fallback"


class TestGovernorProvider:
    def test_governor_applies_provider_to_snapshot(self):
        """Governor applies TrustedContextProvider values to the telemetry snapshot.

        The snapshot passed to enforce_next_step() has low utilisation values
        (score ≈ 0.04, well below the trip threshold).  The provider injects
        high utilisation values (score ≈ 0.84, above the trip threshold).
        The breaker should trip to THROTTLED, proving the provider was applied.
        """
        numerail_ext = pytest.importorskip(
            "numerail_ext",
            reason="numerail_ext not installed; skipping governor provider test",
        )

        import hashlib
        import json
        from time import time_ns
        from typing import Mapping

        from numerail_ext.survivability.breaker import BreakerStateMachine
        from numerail_ext.survivability.governor import StateTransitionGovernor
        from numerail_ext.survivability.local_backend import LocalNumerailBackend
        from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
        from numerail_ext.survivability.types import (
            BreakerMode,
            BreakerThresholds,
            TelemetrySnapshot,
            WorkloadRequest,
        )

        # Provider injects high utilisation — will trip the breaker.
        # score = 0.30*0.99 + 0.25*0.99 + 0.20*0.99 + 0.10*0.99 + 0 = 0.8415
        # With safe_stop=0.99 → THROTTLED (not SAFE_STOP).
        provider = _FixedProvider({
            "current_gpu_util": 0.99,
            "current_api_util": 0.99,
            "current_db_util": 0.99,
            "current_queue_util": 0.99,
            "current_error_rate_pct": 0.0,
        })

        class _Res:
            def acquire(self, *, state_version, expires_at_ns, resource_claims): return "tok"
            def commit(self, *, token, receipt): pass
            def release(self, *, token): pass

        class _Dig:
            def digest(self, payload):
                return hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=str).encode()
                ).hexdigest()

        governor = StateTransitionGovernor(
            backend=LocalNumerailBackend(),
            transition_model=IncidentCommanderTransitionModel(freshness_ns=120_000_000_000),
            reservation_mgr=_Res(),
            digestor=_Dig(),
            thresholds=BreakerThresholds(
                trip_score=0.50, reset_score=0.25, safe_stop_score=0.99
            ),
            bootstrap_budgets={"gpu_shift": 500.0, "external_api_shift": 100.0, "mutation_shift": 40.0},
            trusted_context_provider=provider,
        )

        # Snapshot with LOW utilisation — would NOT trip on its own.
        low_snap = TelemetrySnapshot(
            state_version=1,
            observed_at_ns=time_ns(),
            current_gpu_util=0.05,
            current_api_util=0.05,
            current_db_util=0.05,
            current_queue_util=0.05,
            current_error_rate_pct=0.0,
            ctrl_gpu_reserve_seconds=30.0,
            ctrl_api_reserve_calls=5.0,
            ctrl_parallel_reserve=4.0,
            ctrl_cloud_mutation_reserve=2.0,
            gpu_disturbance_margin_seconds=15.0,
            api_disturbance_margin_calls=3.0,
            db_disturbance_margin_pct=5.0,
            queue_disturbance_margin_pct=3.0,
        )

        request = WorkloadRequest(
            prompt_k=5.0, completion_k=2.0, internal_tool_calls=5.0,
            external_api_calls=3.0, cloud_mutation_calls=1.0, gpu_seconds=10.0,
            parallel_workers=2.0, traffic_shift_pct=0.0, worker_scale_up_pct=0.0,
            feature_flag_changes=0.0, rollback_batch_pct=0.0,
            pager_notifications=1.0, customer_comms_count=0.0,
        )

        step = governor.enforce_next_step(
            request=request, snapshot=low_snap, action_id="gov_provider_test"
        )

        # Provider overwrote the snapshot with high utilisation → breaker trips.
        assert step.breaker.mode != BreakerMode.CLOSED, (
            f"Expected THROTTLED (provider should have tripped the breaker), "
            f"got {step.breaker.mode}. Score: {step.breaker.overload_score:.4f}"
        )
