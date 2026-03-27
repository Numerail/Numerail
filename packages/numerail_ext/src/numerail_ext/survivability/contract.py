"""Numerail Policy Contract — cryptographically structured policy interchange.

A NumerailPolicyContract is a self-contained, content-addressable, immutable
record of a complete enforcement policy.  It captures everything needed to
reproduce an enforcement decision: the constraint geometry, the enforcement
configuration, the trust partition, the budget specifications, and the
cryptographic linkage to the policy's provenance chain.

Design principles:
    1. Content-addressable: the contract's digest is computed from its
       canonical serialization.  Two contracts with identical content
       produce identical digests.  The digest is the contract's identity.
    2. Chain-linked: each contract references its predecessor's digest,
       creating an append-only version chain that can be verified
       independently.  Tampering with any contract invalidates all
       successors.
    3. Separable: the contract separates the policy (what constraints
       exist) from the authority (who authored them and when) from the
       activation (when and where the policy is in effect).  Each can be
       independently verified.
    4. V5-native: the contract's ``v5_config`` field is a complete
       ``NumerailSystem.from_config()``-compatible dict.  The contract
       can be loaded directly into V5 without transformation.
    5. Breaker-compatible: the contract's schema, constraint names,
       trusted fields, budget keys, and dimension policies are
       structurally identical to the breaker suite's policy builder
       output.  The governor can use a contract as its active config.

Cryptographic structure:
    ┌─────────────────────────────────────────────┐
    │  NumerailPolicyContract                     │
    │                                             │
    │  header:                                    │
    │    contract_version: "1.0"                  │
    │    schema_version: "numerail-v5"            │
    │    created_at_ns: int                       │
    │    author_id: str                           │
    │    previous_digest: str | None              │
    │    activation:                              │
    │      policy_id: str                         │
    │      effective_from_ns: int                 │
    │      effective_until_ns: int | None         │
    │      scope: str                             │
    │                                             │
    │  geometry:                                  │
    │    schema: {fields, normalizers, defaults}   │
    │    polytope: {A, b, names, tags}            │
    │    quadratic_constraints: [...]             │
    │    socp_constraints: [...]                  │
    │    psd_constraints: [...]                   │
    │                                             │
    │  trust:                                     │
    │    trusted_fields: [...]                    │
    │    dimension_policies: {...}                │
    │                                             │
    │  enforcement:                               │
    │    mode, routing_thresholds, safety_margin,  │
    │    hard_wall_constraints, solver config      │
    │                                             │
    │  budgets: [...]                             │
    │                                             │
    │  digest: SHA-256(canonical(above))           │
    │  signature: optional external signature     │
    └─────────────────────────────────────────────┘

Usage:
    # Create
    contract = NumerailPolicyContract.from_v5_config(
        config=build_global_default(),
        author_id="governance-council",
        policy_id="global-default::v1.0",
        scope="production",
    )

    # Verify
    assert contract.verify_digest()

    # Chain
    next_contract = NumerailPolicyContract.from_v5_config(
        config=updated_config,
        author_id="governance-council",
        policy_id="global-default::v1.1",
        previous_digest=contract.digest,
    )

    # Load into V5
    system = NumerailSystem.from_config(contract.v5_config)

    # Load into governor
    governor.backend.set_active_config(contract.v5_config)

    # Export
    wire = contract.to_bytes()
    restored = NumerailPolicyContract.from_bytes(wire)
    assert restored.digest == contract.digest
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from time import time_ns
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  CANONICAL SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _canonical_json(obj: Any) -> str:
    """Deterministic JSON for content-addressing.

    Sorted keys, no whitespace, numpy-safe.  Matches V5's
    ``_deterministic_json`` output so that digests computed here
    and inside V5 are interoperable.
    """
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  CONTRACT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContractActivation:
    """When and where this policy is in effect."""
    policy_id: str
    effective_from_ns: int
    effective_until_ns: Optional[int] = None
    scope: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "policy_id": self.policy_id,
            "effective_from_ns": self.effective_from_ns,
            "scope": self.scope,
        }
        if self.effective_until_ns is not None:
            d["effective_until_ns"] = self.effective_until_ns
        return d


@dataclass(frozen=True)
class ContractHeader:
    """Provenance and chain linkage."""
    contract_version: str
    schema_version: str
    created_at_ns: int
    author_id: str
    activation: ContractActivation
    previous_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "created_at_ns": self.created_at_ns,
            "author_id": self.author_id,
            "activation": self.activation.to_dict(),
        }
        if self.previous_digest is not None:
            d["previous_digest"] = self.previous_digest
        return d


@dataclass(frozen=True)
class ContractGeometry:
    """The constraint geometry — everything V5 needs to build a FeasibleRegion."""
    schema: Dict[str, Any]
    polytope: Dict[str, Any]
    quadratic_constraints: List[Dict[str, Any]]
    socp_constraints: List[Dict[str, Any]]
    psd_constraints: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": copy.deepcopy(self.schema),
            "polytope": copy.deepcopy(self.polytope),
            "quadratic_constraints": copy.deepcopy(self.quadratic_constraints),
            "socp_constraints": copy.deepcopy(self.socp_constraints),
            "psd_constraints": copy.deepcopy(self.psd_constraints),
        }

    @property
    def dimension(self) -> int:
        return len(self.schema.get("fields", []))

    @property
    def linear_constraint_count(self) -> int:
        return len(self.polytope.get("names", []))

    @property
    def constraint_summary(self) -> str:
        parts = [f"{self.linear_constraint_count} linear"]
        if self.quadratic_constraints:
            parts.append(f"{len(self.quadratic_constraints)} quadratic")
        if self.socp_constraints:
            parts.append(f"{len(self.socp_constraints)} SOCP")
        if self.psd_constraints:
            parts.append(f"{len(self.psd_constraints)} PSD")
        return ", ".join(parts)


@dataclass(frozen=True)
class ContractTrust:
    """The trust partition — which fields are server-authoritative and how
    each dimension is handled during projection."""
    trusted_fields: Tuple[str, ...]
    dimension_policies: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trusted_fields": list(self.trusted_fields),
            "dimension_policies": dict(self.dimension_policies),
        }

    @property
    def forbidden_fields(self) -> Tuple[str, ...]:
        return tuple(k for k, v in self.dimension_policies.items() if v == "forbidden")

    @property
    def flagged_fields(self) -> Tuple[str, ...]:
        return tuple(k for k, v in self.dimension_policies.items() if v == "project_with_flag")


@dataclass(frozen=True)
class ContractEnforcement:
    """Enforcement configuration — mode, routing, safety margin, solver params."""
    mode: str
    routing_thresholds: Optional[Dict[str, float]]
    safety_margin: float
    hard_wall_constraints: Tuple[str, ...]
    solver_max_iter: int
    solver_tol: float
    dykstra_max_iter: int
    max_distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "mode": self.mode,
            "safety_margin": self.safety_margin,
            "hard_wall_constraints": list(self.hard_wall_constraints),
            "solver_max_iter": self.solver_max_iter,
            "solver_tol": self.solver_tol,
            "dykstra_max_iter": self.dykstra_max_iter,
        }
        if self.routing_thresholds is not None:
            d["routing_thresholds"] = dict(self.routing_thresholds)
        if self.max_distance is not None:
            d["max_distance"] = self.max_distance
        return d


@dataclass(frozen=True)
class ContractBudget:
    """A single budget specification."""
    name: str
    constraint_name: str
    weight: Dict[str, float]
    initial: float
    consumption_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "constraint_name": self.constraint_name,
            "weight": dict(self.weight),
            "initial": self.initial,
            "consumption_mode": self.consumption_mode,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  THE CONTRACT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NumerailPolicyContract:
    """Content-addressable, chain-linked, V5-native policy contract.

    The contract is immutable after construction.  Its digest is computed
    from the canonical serialization of all fields except the digest itself
    and the optional signature.  Two contracts with identical content
    produce identical digests regardless of construction time or order.
    """

    header: ContractHeader
    geometry: ContractGeometry
    trust: ContractTrust
    enforcement: ContractEnforcement
    budgets: Tuple[ContractBudget, ...]
    digest: str
    # NOT CURRENTLY IMPLEMENTED — placeholder for future signing support.
    # The digest field is the sole integrity verification mechanism.
    # Do not rely on this field for compliance or provenance verification.
    signature: Optional[str] = None

    # ── Content-addressable identity ─────────────────────────────────

    def _digestable_dict(self) -> Dict[str, Any]:
        """The dict that is hashed to produce the digest.
        Excludes digest and signature — those are computed from this."""
        return {
            "header": self.header.to_dict(),
            "geometry": self.geometry.to_dict(),
            "trust": self.trust.to_dict(),
            "enforcement": self.enforcement.to_dict(),
            "budgets": [b.to_dict() for b in self.budgets],
        }

    def verify_digest(self) -> bool:
        """Recompute and verify the contract's content digest."""
        expected = _sha256(_canonical_json(self._digestable_dict()))
        return self.digest == expected

    # ── Chain verification ───────────────────────────────────────────

    @staticmethod
    def verify_chain(contracts: Sequence["NumerailPolicyContract"]) -> Tuple[bool, int]:
        """Verify a chain of contracts.  Returns (valid, depth).
        The first contract's previous_digest may be None (genesis)."""
        for i, c in enumerate(contracts):
            if not c.verify_digest():
                return False, i
            if i > 0:
                if c.header.previous_digest != contracts[i - 1].digest:
                    return False, i
        return True, len(contracts)

    # ── V5 config extraction ─────────────────────────────────────────

    @property
    def v5_config(self) -> Dict[str, Any]:
        """Extract a complete NumerailSystem.from_config()-compatible dict.

        This is the primary integration point.  The returned dict can be
        passed directly to V5 or to the breaker suite's governor.
        """
        ec = self.enforcement.to_dict()
        ec["dimension_policies"] = dict(self.trust.dimension_policies)

        config: Dict[str, Any] = {
            "schema": dict(self.geometry.schema),
            "polytope": dict(self.geometry.polytope),
            "enforcement": ec,
            "trusted_fields": list(self.trust.trusted_fields),
            "budgets": [b.to_dict() for b in self.budgets],
        }
        if self.geometry.quadratic_constraints:
            config["quadratic_constraints"] = [
                dict(qc) for qc in self.geometry.quadratic_constraints
            ]
        if self.geometry.socp_constraints:
            config["socp_constraints"] = [
                dict(sc) for sc in self.geometry.socp_constraints
            ]
        if self.geometry.psd_constraints:
            config["psd_constraints"] = [
                dict(pc) for pc in self.geometry.psd_constraints
            ]
        return config

    # ── Wire format ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Full serializable representation including digest and signature."""
        d = self._digestable_dict()
        d["digest"] = self.digest
        if self.signature is not None:
            d["signature"] = self.signature
        return d

    def to_json(self) -> str:
        """Canonical JSON serialization."""
        return _canonical_json(self.to_dict())

    def to_bytes(self) -> bytes:
        """UTF-8 encoded canonical JSON."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NumerailPolicyContract":
        """Reconstruct from a serialized dict.  Recomputes and verifies digest."""
        h = d["header"]
        act = h["activation"]
        header = ContractHeader(
            contract_version=h["contract_version"],
            schema_version=h["schema_version"],
            created_at_ns=h["created_at_ns"],
            author_id=h["author_id"],
            activation=ContractActivation(
                policy_id=act["policy_id"],
                effective_from_ns=act["effective_from_ns"],
                effective_until_ns=act.get("effective_until_ns"),
                scope=act.get("scope", "default"),
            ),
            previous_digest=h.get("previous_digest"),
        )

        g = d["geometry"]
        geometry = ContractGeometry(
            schema=g["schema"],
            polytope=g["polytope"],
            quadratic_constraints=g.get("quadratic_constraints", []),
            socp_constraints=g.get("socp_constraints", []),
            psd_constraints=g.get("psd_constraints", []),
        )

        t = d["trust"]
        trust = ContractTrust(
            trusted_fields=tuple(t["trusted_fields"]),
            dimension_policies=dict(t["dimension_policies"]),
        )

        e = d["enforcement"]
        enforcement = ContractEnforcement(
            mode=e["mode"],
            routing_thresholds=e.get("routing_thresholds"),
            safety_margin=e["safety_margin"],
            hard_wall_constraints=tuple(e.get("hard_wall_constraints", [])),
            solver_max_iter=e.get("solver_max_iter", 2000),
            solver_tol=e.get("solver_tol", 1e-6),
            dykstra_max_iter=e.get("dykstra_max_iter", 10000),
            max_distance=e.get("max_distance"),
        )

        budgets = tuple(
            ContractBudget(
                name=b["name"],
                constraint_name=b["constraint_name"],
                weight=dict(b["weight"]),
                initial=b["initial"],
                consumption_mode=b["consumption_mode"],
            )
            for b in d.get("budgets", [])
        )

        contract = cls(
            header=header,
            geometry=geometry,
            trust=trust,
            enforcement=enforcement,
            budgets=budgets,
            digest=d["digest"],
            signature=d.get("signature"),
        )

        if not contract.verify_digest():
            raise ValueError(
                f"Contract digest verification failed. "
                f"Stored: {d['digest'][:16]}..., "
                f"Computed: {_sha256(_canonical_json(contract._digestable_dict()))[:16]}..."
            )

        return contract

    @classmethod
    def from_json(cls, s: str) -> "NumerailPolicyContract":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_bytes(cls, b: bytes) -> "NumerailPolicyContract":
        return cls.from_json(b.decode("utf-8"))

    # ── Factory from V5 config ───────────────────────────────────────

    @classmethod
    def from_v5_config(
        cls,
        config: Dict[str, Any],
        *,
        author_id: str,
        policy_id: Optional[str] = None,
        scope: str = "default",
        effective_from_ns: Optional[int] = None,
        effective_until_ns: Optional[int] = None,
        previous_digest: Optional[str] = None,
        signature: Optional[str] = None,
    ) -> "NumerailPolicyContract":
        """Construct a contract from a V5-compatible config dict.

        This is the primary construction path.  Takes a config dict
        (from build_global_default, build_v5_policy_from_envelope, or
        any hand-written V5 config) and wraps it in a content-addressable
        contract with provenance metadata.
        """
        now = time_ns()
        pid = policy_id or config.get("policy_id", "unnamed")

        header = ContractHeader(
            contract_version="1.0",
            schema_version="numerail-v5",
            created_at_ns=now,
            author_id=author_id,
            activation=ContractActivation(
                policy_id=pid,
                effective_from_ns=effective_from_ns or now,
                effective_until_ns=effective_until_ns,
                scope=scope,
            ),
            previous_digest=previous_digest,
        )

        sc = config.get("action_schema", config.get("schema", {}))
        geometry = ContractGeometry(
            schema=dict(sc),
            polytope=dict(config.get("polytope", {})),
            quadratic_constraints=[dict(qc) for qc in config.get("quadratic_constraints", [])],
            socp_constraints=[dict(sc_) for sc_ in config.get("socp_constraints", [])],
            psd_constraints=[dict(pc) for pc in config.get("psd_constraints", [])],
        )

        ec = config.get("enforcement", {})
        trust = ContractTrust(
            trusted_fields=tuple(config.get("trusted_fields", [])),
            dimension_policies=dict(ec.get("dimension_policies", {})),
        )

        rt = ec.get("routing_thresholds")
        enforcement = ContractEnforcement(
            mode=ec.get("mode", "project"),
            routing_thresholds=dict(rt) if rt else None,
            safety_margin=float(ec.get("safety_margin", 1.0)),
            hard_wall_constraints=tuple(ec.get("hard_wall_constraints", [])),
            solver_max_iter=int(ec.get("solver_max_iter", 2000)),
            solver_tol=float(ec.get("solver_tol", 1e-6)),
            dykstra_max_iter=int(ec.get("dykstra_max_iter", 10000)),
            max_distance=ec.get("max_distance"),
        )

        budgets = tuple(
            ContractBudget(
                name=b["name"],
                constraint_name=b["constraint_name"],
                weight=dict(b["weight"]) if isinstance(b.get("weight"), dict) else {b.get("dimension_name", ""): float(b.get("weight", 1.0))},
                initial=float(b["initial"]),
                consumption_mode=b.get("consumption_mode", b.get("mode", "nonnegative")),
            )
            for b in config.get("budgets", [])
        )

        # Compute content digest
        provisional = cls(
            header=header, geometry=geometry, trust=trust,
            enforcement=enforcement, budgets=budgets,
            digest="",  # placeholder
            signature=None,
        )
        digest = _sha256(_canonical_json(provisional._digestable_dict()))

        return cls(
            header=header, geometry=geometry, trust=trust,
            enforcement=enforcement, budgets=budgets,
            digest=digest, signature=signature,
        )

    # ── Introspection ────────────────────────────────────────────────

    @property
    def policy_id(self) -> str:
        return self.header.activation.policy_id

    @property
    def author(self) -> str:
        return self.header.author_id

    @property
    def dimension(self) -> int:
        return self.geometry.dimension

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"NumerailPolicyContract [{self.digest[:12]}...]",
            f"  Policy:      {self.policy_id}",
            f"  Author:      {self.author}",
            f"  Scope:       {self.header.activation.scope}",
            f"  Schema:      {self.dimension} dimensions",
            f"  Constraints: {self.geometry.constraint_summary}",
            f"  Trust:       {len(self.trust.trusted_fields)} trusted, "
            f"{len(self.trust.forbidden_fields)} forbidden, "
            f"{len(self.trust.flagged_fields)} flagged",
            f"  Budgets:     {len(self.budgets)}",
            f"  Mode:        {self.enforcement.mode}",
            f"  Margin:      {self.enforcement.safety_margin}",
            f"  Chain:       {'genesis' if self.header.previous_digest is None else self.header.previous_digest[:12] + '...'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"NumerailPolicyContract("
            f"policy_id={self.policy_id!r}, "
            f"digest={self.digest[:12]}..., "
            f"dim={self.dimension})"
        )
