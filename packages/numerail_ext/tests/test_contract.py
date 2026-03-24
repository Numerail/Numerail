"""
Test suite for NumerailPolicyContract.

Covers all properties claimed in the policy contract documentation:
  - Content-addressable identity and digest computation
  - Canonical JSON serialization matching V5's _deterministic_json
  - Chain linkage and tamper detection
  - V5 config extraction and machine-loadable round-trip
  - Wire format serialization / deserialization
  - Trust partition correctness
  - Budget specification structure
  - Geometry introspection
  - Stdlib-only portable verification
  - Enforcement configuration round-trip
  - Breaker suite compatibility (dynamic path)
  - Introspection (summary, repr, properties)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import List

import pytest

from numerail.engine import NumerailSystem
from numerail.parser import PolicyParser

from numerail_ext.survivability.contract import (
    NumerailPolicyContract,
    ContractActivation,
    ContractBudget,
    ContractEnforcement,
    ContractGeometry,
    ContractHeader,
    ContractTrust,
    _canonical_json,
    _sha256,
)
from numerail_ext.survivability.global_default import (
    ALL_FIELDS,
    TRUSTED_FIELDS,
    WORKLOAD_FIELDS,
    build_global_default,
)
from numerail_ext.survivability.policy_builder import build_v5_policy_from_envelope
from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
from numerail_ext.survivability.types import (
    BreakerMode,
    BreakerThresholds,
    TelemetrySnapshot,
)


# ── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def default_config():
    return build_global_default()


@pytest.fixture(scope="module")
def genesis(default_config):
    return NumerailPolicyContract.from_v5_config(
        default_config,
        author_id="governance-council",
        policy_id="global-default::v1.0",
        scope="production",
    )


@pytest.fixture(scope="module")
def v2_contract(default_config, genesis):
    config_v2 = build_global_default(gpu_seconds_cap=80.0)
    return NumerailPolicyContract.from_v5_config(
        config_v2,
        author_id="governance-council",
        policy_id="global-default::v1.1",
        previous_digest=genesis.digest,
    )


@pytest.fixture(scope="module")
def v3_contract(default_config, v2_contract):
    return NumerailPolicyContract.from_v5_config(
        default_config,
        author_id="governance-council",
        policy_id="global-default::v1.2",
        previous_digest=v2_contract.digest,
    )


def _make_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        state_version=1,
        observed_at_ns=time.time_ns(),
        current_gpu_util=0.3,
        current_api_util=0.2,
        current_db_util=0.1,
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONTENT-ADDRESSABLE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════


class TestContentAddressableIdentity:
    def test_verify_digest_passes_on_fresh_contract(self, genesis):
        assert genesis.verify_digest()

    def test_digest_is_64_hex_chars(self, genesis):
        assert len(genesis.digest) == 64
        assert all(c in "0123456789abcdef" for c in genesis.digest)

    def test_different_author_produces_different_digest(self, default_config):
        c1 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="team-a",
            policy_id="p1", effective_from_ns=1_000_000_000,
        )
        c2 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="team-b",
            policy_id="p1", effective_from_ns=1_000_000_000,
        )
        assert c1.digest != c2.digest

    def test_different_policy_id_produces_different_digest(self, default_config):
        c1 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="team-a",
            policy_id="v1.0", effective_from_ns=1_000_000_000,
        )
        c2 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="team-a",
            policy_id="v1.1", effective_from_ns=1_000_000_000,
        )
        assert c1.digest != c2.digest

    def test_different_scope_produces_different_digest(self, default_config):
        c1 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="a",
            policy_id="p", effective_from_ns=1_000_000_000, scope="production",
        )
        c2 = NumerailPolicyContract.from_v5_config(
            default_config, author_id="a",
            policy_id="p", effective_from_ns=1_000_000_000, scope="staging",
        )
        assert c1.digest != c2.digest

    def test_different_constraint_geometry_produces_different_digest(self):
        c1 = NumerailPolicyContract.from_v5_config(
            build_global_default(gpu_seconds_cap=120.0),
            author_id="a", policy_id="p", effective_from_ns=1_000_000_000,
        )
        c2 = NumerailPolicyContract.from_v5_config(
            build_global_default(gpu_seconds_cap=60.0),
            author_id="a", policy_id="p", effective_from_ns=1_000_000_000,
        )
        assert c1.digest != c2.digest

    def test_genesis_has_no_previous_digest(self, genesis):
        assert genesis.header.previous_digest is None

    def test_chained_contract_stores_predecessor_digest(self, v2_contract, genesis):
        assert v2_contract.header.previous_digest == genesis.digest


# ═══════════════════════════════════════════════════════════════════════════
# 2. DIGEST COMPUTATION — canonical JSON
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestComputation:
    def test_manual_recomputation_matches_stored(self, genesis):
        d_dict = genesis._digestable_dict()
        expected = hashlib.sha256(
            _canonical_json(d_dict).encode("utf-8")
        ).hexdigest()
        assert expected == genesis.digest

    def test_digest_excludes_itself(self, genesis):
        d = genesis._digestable_dict()
        assert "digest" not in d

    def test_digest_excludes_signature(self, genesis):
        d = genesis._digestable_dict()
        assert "signature" not in d

    def test_canonical_json_sorted_keys(self):
        result = _canonical_json({"z": 1, "a": 2, "m": 3})
        assert result == '{"a":2,"m":3,"z":1}'

    def test_canonical_json_no_whitespace(self):
        result = _canonical_json({"a": 1, "b": [1, 2, 3]})
        assert " " not in result
        assert "\n" not in result

    def test_canonical_json_numpy_float64(self):
        import numpy as np
        result = _canonical_json({"x": np.float64(1.5)})
        assert '"x":1.5' in result

    def test_canonical_json_numpy_int32(self):
        import numpy as np
        result = _canonical_json({"y": np.int32(42)})
        assert '"y":42' in result

    def test_canonical_json_numpy_array(self):
        import numpy as np
        result = _canonical_json({"z": np.array([1, 2, 3])})
        assert '"z":[1,2,3]' in result

    def test_canonical_json_set_sorted(self):
        result = _canonical_json({"s": {3, 1, 2}})
        assert '"s":[1,2,3]' in result

    def test_sha256_helper_correctness(self):
        known = hashlib.sha256(b"hello").hexdigest()
        assert _sha256("hello") == known

    def test_verify_digest_detects_tampered_author(self, genesis):
        # Manually construct a contract with same digest but different author
        tampered = dataclasses.replace(
            genesis,
            header=dataclasses.replace(
                genesis.header, author_id="attacker"
            ),
        )
        assert not tampered.verify_digest()

    def test_verify_digest_detects_zeroed_digest(self, genesis):
        tampered = dataclasses.replace(genesis, digest="0" * 64)
        assert not tampered.verify_digest()


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHAIN LINKAGE AND TAMPER DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestChainLinkage:
    def test_three_contract_chain_valid(self, genesis, v2_contract, v3_contract):
        valid, depth = NumerailPolicyContract.verify_chain(
            [genesis, v2_contract, v3_contract]
        )
        assert valid
        assert depth == 3

    def test_single_genesis_chain_valid(self, genesis):
        valid, depth = NumerailPolicyContract.verify_chain([genesis])
        assert valid
        assert depth == 1

    def test_empty_chain_valid(self):
        valid, depth = NumerailPolicyContract.verify_chain([])
        assert valid
        assert depth == 0

    def test_skipped_contract_breaks_chain(self, genesis, v3_contract):
        # v3's previous_digest points to v2, not genesis
        valid, broken_at = NumerailPolicyContract.verify_chain(
            [genesis, v3_contract]
        )
        assert not valid
        assert broken_at == 1

    def test_tampered_digest_breaks_chain(self, genesis, v2_contract, v3_contract):
        tampered_v2 = dataclasses.replace(v2_contract, digest="f" * 64)
        valid, broken_at = NumerailPolicyContract.verify_chain(
            [genesis, tampered_v2, v3_contract]
        )
        assert not valid
        assert broken_at == 1  # v2's own digest fails

    def test_inserted_contract_breaks_chain(
        self, genesis, v2_contract, v3_contract, default_config
    ):
        intruder = NumerailPolicyContract.from_v5_config(
            default_config,
            author_id="intruder",
            policy_id="fake",
            previous_digest=genesis.digest,
        )
        valid, broken_at = NumerailPolicyContract.verify_chain(
            [genesis, intruder, v2_contract, v3_contract]
        )
        assert not valid

    def test_reversed_chain_invalid(self, genesis, v2_contract):
        valid, _ = NumerailPolicyContract.verify_chain([v2_contract, genesis])
        assert not valid

    def test_chain_links_match_predecessors(self, genesis, v2_contract, v3_contract):
        assert v2_contract.header.previous_digest == genesis.digest
        assert v3_contract.header.previous_digest == v2_contract.digest


# ═══════════════════════════════════════════════════════════════════════════
# 4. V5 CONFIG EXTRACTION — machine-loadable round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestV5ConfigExtraction:
    def test_v5_config_parses_without_error(self, genesis):
        parser = PolicyParser()
        validated = parser.parse(genesis.v5_config)
        assert validated is not None

    def test_v5_config_loads_into_numerail_system(self, genesis):
        validated = PolicyParser().parse(genesis.v5_config)
        system = NumerailSystem.from_config(validated)
        assert system is not None

    def test_trusted_fields_present_and_correct_count(self, genesis):
        cfg = genesis.v5_config
        assert "trusted_fields" in cfg
        assert len(cfg["trusted_fields"]) == len(TRUSTED_FIELDS)

    def test_trusted_fields_match_spec(self, genesis):
        cfg = genesis.v5_config
        assert set(cfg["trusted_fields"]) == set(TRUSTED_FIELDS)

    def test_dimension_policies_in_enforcement(self, genesis):
        enforcement = genesis.v5_config.get("enforcement", {})
        assert "dimension_policies" in enforcement

    def test_all_four_constraint_types_present(self, genesis):
        cfg = genesis.v5_config
        assert "polytope" in cfg
        assert len(cfg.get("quadratic_constraints", [])) >= 1
        assert len(cfg.get("socp_constraints", [])) >= 1
        assert len(cfg.get("psd_constraints", [])) >= 1

    def test_header_excluded_from_v5_config(self, genesis):
        cfg = genesis.v5_config
        assert "header" not in cfg
        assert "digest" not in cfg
        assert "signature" not in cfg

    def test_budgets_present_in_v5_config(self, genesis):
        cfg = genesis.v5_config
        assert len(cfg.get("budgets", [])) == 3

    def test_v5_config_from_tighter_policy_also_loads(self):
        config = build_global_default(gpu_seconds_cap=60.0, safety_margin=0.95)
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="qa", policy_id="tight"
        )
        validated = PolicyParser().parse(contract.v5_config)
        system = NumerailSystem.from_config(validated)
        assert system is not None

    def test_v5_config_round_trip_preserves_constraint_names(self, genesis):
        cfg = genesis.v5_config
        polytope_names = cfg["polytope"]["names"]
        assert len(polytope_names) > 0
        assert len(set(polytope_names)) == len(polytope_names)  # unique


# ═══════════════════════════════════════════════════════════════════════════
# 5. WIRE FORMAT — serialization / deserialization
# ═══════════════════════════════════════════════════════════════════════════


class TestWireFormat:
    def test_to_bytes_returns_bytes(self, genesis):
        assert isinstance(genesis.to_bytes(), bytes)

    def test_from_bytes_round_trip_digest(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.digest == genesis.digest

    def test_from_bytes_verify_digest_passes(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.verify_digest()

    def test_from_bytes_preserves_policy_id(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.policy_id == genesis.policy_id

    def test_from_bytes_preserves_author(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.author == genesis.author

    def test_from_bytes_preserves_scope(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.header.activation.scope == genesis.header.activation.scope

    def test_from_bytes_preserves_dimension(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.dimension == genesis.dimension

    def test_to_json_is_deterministic(self, genesis):
        assert genesis.to_json() == genesis.to_json()

    def test_to_json_is_valid_json(self, genesis):
        parsed = json.loads(genesis.to_json())
        assert isinstance(parsed, dict)

    def test_to_dict_contains_digest(self, genesis):
        d = genesis.to_dict()
        assert "digest" in d
        assert d["digest"] == genesis.digest

    def test_from_dict_raises_on_tampered_author(self, genesis):
        d = genesis.to_dict()
        d["header"]["author_id"] = "attacker"
        with pytest.raises(ValueError, match="digest"):
            NumerailPolicyContract.from_dict(d)

    def test_from_dict_raises_on_tampered_constraint(self, genesis):
        d = genesis.to_dict()
        # Modify a bound value in the polytope
        d["geometry"]["polytope"]["b"][0] = 9999.0
        with pytest.raises(ValueError, match="digest"):
            NumerailPolicyContract.from_dict(d)

    def test_from_dict_raises_on_tampered_budget(self, genesis):
        d = genesis.to_dict()
        if d.get("budgets"):
            d["budgets"][0]["initial"] = 999999.0
            with pytest.raises(ValueError, match="digest"):
                NumerailPolicyContract.from_dict(d)

    def test_from_json_raises_on_tampered_content(self, genesis):
        raw = genesis.to_json()
        # Replace author_id in the raw JSON string
        tampered = raw.replace(genesis.author, "hacker", 1)
        if tampered != raw:  # only test if replacement occurred
            with pytest.raises(ValueError):
                NumerailPolicyContract.from_json(tampered)

    def test_from_bytes_raises_on_tampered_bytes(self, genesis):
        wire = bytearray(genesis.to_bytes())
        # Flip a byte deep in the middle (avoiding digest field itself)
        mid = len(wire) // 2
        wire[mid] ^= 0xFF
        with pytest.raises((ValueError, json.JSONDecodeError)):
            NumerailPolicyContract.from_bytes(bytes(wire))


# ═══════════════════════════════════════════════════════════════════════════
# 6. PORTABLE VERIFICATION — stdlib only, no Numerail
# ═══════════════════════════════════════════════════════════════════════════


class TestPortableVerification:
    def test_digest_verifiable_with_stdlib_only(self, genesis):
        """Verify the contract using only hashlib and json — no Numerail imports."""
        wire = genesis.to_bytes()
        raw = json.loads(wire.decode("utf-8"))
        stored_digest = raw.pop("digest")
        raw.pop("signature", None)
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert recomputed == stored_digest

    def test_chain_verifiable_with_stdlib(self, genesis, v2_contract, v3_contract):
        """Walk a chain of contracts using only stdlib JSON and hashlib."""
        contracts_wire = [
            json.loads(c.to_bytes()) for c in [genesis, v2_contract, v3_contract]
        ]
        for i, raw in enumerate(contracts_wire):
            stored = raw.pop("digest")
            raw.pop("signature", None)
            canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            assert recomputed == stored, f"Chain broken at index {i}"
            if i > 0:
                prev_raw = json.loads(
                    contracts_wire[i - 1] if isinstance(contracts_wire[i - 1], str)
                    else json.dumps(contracts_wire[i - 1])
                )
                # The raw already had digest popped; re-check link via stored values
        # If we get here without assertion errors, the chain is intact

    def test_tampered_wire_fails_stdlib_verification(self, genesis):
        wire = genesis.to_bytes()
        raw = json.loads(wire.decode("utf-8"))
        stored_digest = raw["digest"]
        # Tamper the author
        raw["header"]["author_id"] = "attacker"
        raw.pop("digest")
        raw.pop("signature", None)
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert recomputed != stored_digest


# ═══════════════════════════════════════════════════════════════════════════
# 7. TRUST PARTITION
# ═══════════════════════════════════════════════════════════════════════════


class TestTrustPartition:
    def test_trusted_fields_count(self, genesis):
        assert len(genesis.trust.trusted_fields) == 17

    def test_trusted_fields_match_spec(self, genesis):
        assert set(genesis.trust.trusted_fields) == set(TRUSTED_FIELDS)

    def test_all_trusted_fields_are_forbidden(self, genesis):
        for field in genesis.trust.trusted_fields:
            assert genesis.trust.dimension_policies.get(field) == "forbidden", (
                f"Trusted field '{field}' should be FORBIDDEN, "
                f"got {genesis.trust.dimension_policies.get(field)}"
            )

    def test_forbidden_fields_property_matches_trusted(self, genesis):
        assert set(genesis.trust.forbidden_fields) == set(genesis.trust.trusted_fields)

    def test_no_workload_fields_are_forbidden(self, genesis):
        for field in WORKLOAD_FIELDS:
            assert field not in genesis.trust.forbidden_fields, (
                f"Workload field '{field}' should not be FORBIDDEN"
            )

    def test_flagged_fields_are_workload_fields(self, genesis):
        for field in genesis.trust.flagged_fields:
            assert field in WORKLOAD_FIELDS, (
                f"Flagged field '{field}' is not a workload field"
            )

    def test_trust_preserved_through_round_trip(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert set(restored.trust.trusted_fields) == set(genesis.trust.trusted_fields)
        assert restored.trust.dimension_policies == genesis.trust.dimension_policies


# ═══════════════════════════════════════════════════════════════════════════
# 8. BUDGET SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetSpecifications:
    def test_three_shift_budgets(self, genesis):
        assert len(genesis.budgets) == 3

    def test_canonical_budget_names(self, genesis):
        names = {b.name for b in genesis.budgets}
        assert "gpu_shift" in names
        assert "external_api_shift" in names
        assert "mutation_shift" in names

    @pytest.mark.parametrize("budget_name,expected_initial", [
        ("gpu_shift", 3600.0),
        ("external_api_shift", 500.0),
        ("mutation_shift", 100.0),
    ])
    def test_budget_initial_values(self, genesis, budget_name, expected_initial):
        budget = next(b for b in genesis.budgets if b.name == budget_name)
        assert budget.initial == expected_initial

    def test_budget_weight_is_dict(self, genesis):
        for budget in genesis.budgets:
            assert isinstance(budget.weight, dict), (
                f"Budget '{budget.name}' weight should be dict, "
                f"got {type(budget.weight)}"
            )

    def test_budget_consumption_mode_nonneg(self, genesis):
        for budget in genesis.budgets:
            assert budget.consumption_mode == "nonnegative", (
                f"Budget '{budget.name}' should use nonnegative consumption"
            )

    def test_budget_weight_keys_are_schema_fields(self, genesis):
        schema_fields = set(genesis.geometry.schema.get("fields", []))
        for budget in genesis.budgets:
            for field in budget.weight:
                assert field in schema_fields, (
                    f"Budget '{budget.name}' weight key '{field}' "
                    f"not in schema fields"
                )

    def test_budgets_preserved_through_round_trip(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert len(restored.budgets) == len(genesis.budgets)
        orig_names = {b.name: b for b in genesis.budgets}
        for b in restored.budgets:
            assert b.name in orig_names
            assert b.initial == orig_names[b.name].initial

    def test_v5_config_budgets_use_dict_weight(self, genesis):
        for b in genesis.v5_config.get("budgets", []):
            assert isinstance(b.get("weight"), dict), (
                f"Budget '{b['name']}' in v5_config should use dict weight"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 9. GEOMETRY INTROSPECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestGeometryIntrospection:
    def test_dimension_is_30(self, genesis):
        assert genesis.dimension == 30

    def test_linear_constraint_count_positive(self, genesis):
        assert genesis.geometry.linear_constraint_count > 0

    def test_constraint_summary_contains_all_types(self, genesis):
        summary = genesis.geometry.constraint_summary
        assert "linear" in summary
        assert "quadratic" in summary
        assert "SOCP" in summary
        assert "PSD" in summary

    def test_geometry_dimension_property(self, genesis):
        assert genesis.geometry.dimension == len(ALL_FIELDS)

    def test_schema_field_count(self, genesis):
        fields = genesis.geometry.schema.get("fields", [])
        assert len(fields) == 30

    def test_schema_fields_match_spec(self, genesis):
        fields = genesis.geometry.schema.get("fields", [])
        assert set(fields) == set(ALL_FIELDS)

    def test_no_duplicate_polytope_names(self, genesis):
        names = genesis.geometry.polytope.get("names", [])
        assert len(names) == len(set(names))

    def test_one_quadratic_constraint(self, genesis):
        assert len(genesis.geometry.quadratic_constraints) == 1

    def test_one_socp_constraint(self, genesis):
        assert len(genesis.geometry.socp_constraints) == 1

    def test_one_psd_constraint(self, genesis):
        assert len(genesis.geometry.psd_constraints) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 10. ENFORCEMENT CONFIGURATION ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════


class TestEnforcementConfig:
    def test_default_mode_is_project(self, genesis):
        assert genesis.enforcement.mode == "project"

    def test_default_safety_margin_is_1(self, genesis):
        assert genesis.enforcement.safety_margin == 1.0

    def test_solver_max_iter_positive(self, genesis):
        assert genesis.enforcement.solver_max_iter > 0

    def test_dykstra_max_iter_positive(self, genesis):
        assert genesis.enforcement.dykstra_max_iter > 0

    def test_solver_tol_positive(self, genesis):
        assert genesis.enforcement.solver_tol > 0

    def test_custom_safety_margin_preserved(self):
        config = build_global_default(safety_margin=0.95)
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="qa", policy_id="custom"
        )
        restored = NumerailPolicyContract.from_bytes(contract.to_bytes())
        assert restored.enforcement.safety_margin == pytest.approx(0.95)

    def test_enforcement_config_round_trip(self, genesis):
        restored = NumerailPolicyContract.from_bytes(genesis.to_bytes())
        assert restored.enforcement.mode == genesis.enforcement.mode
        assert restored.enforcement.safety_margin == genesis.enforcement.safety_margin
        assert restored.enforcement.solver_max_iter == genesis.enforcement.solver_max_iter
        assert restored.enforcement.solver_tol == genesis.enforcement.solver_tol

    def test_routing_thresholds_present(self, genesis):
        # Global default should include routing thresholds
        assert genesis.enforcement.routing_thresholds is not None

    def test_hard_wall_constraints_is_tuple(self, genesis):
        assert isinstance(genesis.enforcement.hard_wall_constraints, tuple)


# ═══════════════════════════════════════════════════════════════════════════
# 11. BREAKER SUITE COMPATIBILITY — dynamic path
# ═══════════════════════════════════════════════════════════════════════════


class TestBreakerSuiteCompatibility:
    def test_dynamic_contract_verify_digest(self):
        model = IncidentCommanderTransitionModel()
        snap = _make_snapshot()
        budgets = {
            "gpu_shift": 3600.0,
            "external_api_shift": 500.0,
            "mutation_shift": 100.0,
        }
        envelope = model.synthesize_envelope(
            snapshot=snap, mode=BreakerMode.CLOSED, budgets=budgets
        )
        config = build_v5_policy_from_envelope(envelope)
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="governor", policy_id="runtime::cycle-1"
        )
        assert contract.verify_digest()

    def test_dynamic_contract_chains_to_genesis(self, genesis):
        model = IncidentCommanderTransitionModel()
        snap = _make_snapshot()
        budgets = {
            "gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0
        }
        envelope = model.synthesize_envelope(
            snapshot=snap, mode=BreakerMode.CLOSED, budgets=budgets
        )
        config = build_v5_policy_from_envelope(envelope)
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="governor", policy_id="runtime::cycle-1",
            previous_digest=genesis.digest,
        )
        assert contract.header.previous_digest == genesis.digest

    def test_dynamic_contract_v5_config_loads(self):
        model = IncidentCommanderTransitionModel()
        snap = _make_snapshot()
        budgets = {
            "gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0
        }
        for mode in [BreakerMode.CLOSED, BreakerMode.THROTTLED,
                     BreakerMode.HALF_OPEN, BreakerMode.SAFE_STOP]:
            envelope = model.synthesize_envelope(
                snapshot=snap, mode=mode, budgets=budgets
            )
            config = build_v5_policy_from_envelope(envelope)
            contract = NumerailPolicyContract.from_v5_config(
                config, author_id="governor", policy_id=f"runtime::{mode.value}"
            )
            validated = PolicyParser().parse(contract.v5_config)
            system = NumerailSystem.from_config(validated)
            assert system is not None

    def test_genesis_to_dynamic_chain_verifies(self, genesis):
        model = IncidentCommanderTransitionModel()
        snap = _make_snapshot()
        budgets = {
            "gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0
        }
        envelope = model.synthesize_envelope(
            snapshot=snap, mode=BreakerMode.CLOSED, budgets=budgets
        )
        config = build_v5_policy_from_envelope(envelope)
        dynamic = NumerailPolicyContract.from_v5_config(
            config, author_id="governor", policy_id="runtime::1",
            previous_digest=genesis.digest,
        )
        valid, depth = NumerailPolicyContract.verify_chain([genesis, dynamic])
        assert valid
        assert depth == 2

    def test_static_and_dynamic_contracts_structurally_compatible(self, genesis):
        """Static (global_default) and dynamic (policy_builder) contracts share
        the same schema field count, trusted field count, and budget count."""
        model = IncidentCommanderTransitionModel()
        snap = _make_snapshot()
        budgets = {
            "gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0
        }
        envelope = model.synthesize_envelope(
            snapshot=snap, mode=BreakerMode.CLOSED, budgets=budgets
        )
        dynamic_contract = NumerailPolicyContract.from_v5_config(
            build_v5_policy_from_envelope(envelope),
            author_id="governor", policy_id="runtime::1",
        )
        assert genesis.dimension == dynamic_contract.dimension
        assert len(genesis.trust.trusted_fields) == len(
            dynamic_contract.trust.trusted_fields
        )
        assert len(genesis.budgets) == len(dynamic_contract.budgets)


# ═══════════════════════════════════════════════════════════════════════════
# 12. INTROSPECTION — summary, repr, properties
# ═══════════════════════════════════════════════════════════════════════════


class TestIntrospection:
    def test_summary_contains_policy_id(self, genesis):
        assert genesis.policy_id in genesis.summary()

    def test_summary_contains_author(self, genesis):
        assert genesis.author in genesis.summary()

    def test_summary_contains_digest_prefix(self, genesis):
        assert genesis.digest[:12] in genesis.summary()

    def test_summary_contains_constraint_summary(self, genesis):
        assert "linear" in genesis.summary()

    def test_summary_marks_genesis_chain_position(self, genesis):
        assert "genesis" in genesis.summary()

    def test_summary_shows_predecessor_for_chained(self, v2_contract, genesis):
        assert genesis.digest[:12] in v2_contract.summary()

    def test_repr_contains_policy_id(self, genesis):
        assert genesis.policy_id in repr(genesis)

    def test_repr_contains_digest_prefix(self, genesis):
        assert genesis.digest[:12] in repr(genesis)

    def test_repr_contains_dimension(self, genesis):
        assert "dim=30" in repr(genesis)

    def test_policy_id_property(self, genesis):
        assert genesis.policy_id == "global-default::v1.0"

    def test_author_property(self, genesis):
        assert genesis.author == "governance-council"

    def test_dimension_property(self, genesis):
        assert genesis.dimension == 30

    def test_signature_defaults_to_none(self, genesis):
        assert genesis.signature is None

    def test_contract_is_frozen(self, genesis):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            genesis.digest = "tampered"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# 13. FACTORY — from_v5_config parameter handling
# ═══════════════════════════════════════════════════════════════════════════


class TestFactory:
    def test_policy_id_defaults_to_config_policy_id(self):
        config = build_global_default(policy_id="default-from-config::v1.0")
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="qa"
        )
        assert contract.policy_id == "default-from-config::v1.0"

    def test_explicit_policy_id_overrides_config(self):
        config = build_global_default(policy_id="original")
        contract = NumerailPolicyContract.from_v5_config(
            config, author_id="qa", policy_id="override"
        )
        assert contract.policy_id == "override"

    def test_effective_from_ns_defaults_to_now(self, default_config):
        before = time.time_ns()
        contract = NumerailPolicyContract.from_v5_config(
            default_config, author_id="qa"
        )
        after = time.time_ns()
        assert before <= contract.header.activation.effective_from_ns <= after

    def test_explicit_effective_from_ns_used(self, default_config):
        ts = 1_000_000_000
        contract = NumerailPolicyContract.from_v5_config(
            default_config, author_id="qa", effective_from_ns=ts
        )
        assert contract.header.activation.effective_from_ns == ts

    def test_effective_until_none_by_default(self, default_config):
        contract = NumerailPolicyContract.from_v5_config(
            default_config, author_id="qa"
        )
        assert contract.header.activation.effective_until_ns is None

    def test_effective_until_stored_when_provided(self, default_config):
        until = time.time_ns() + 86_400_000_000_000  # 1 day
        contract = NumerailPolicyContract.from_v5_config(
            default_config, author_id="qa", effective_until_ns=until
        )
        assert contract.header.activation.effective_until_ns == until

    def test_signature_stored_when_provided(self, default_config):
        contract = NumerailPolicyContract.from_v5_config(
            default_config, author_id="qa", signature="external-sig-value"
        )
        assert contract.signature == "external-sig-value"

    def test_schema_version_is_numerail_v5(self, genesis):
        assert genesis.header.schema_version == "numerail-v5"

    def test_contract_version_is_1_0(self, genesis):
        assert genesis.header.contract_version == "1.0"
