# Technical Reference

---

## Installation

```bash
pip install numpy scipy
```

```python
import numerail as nm
```

All classes and functions are available at the module's top level. Dependencies: numpy ≥ 1.21, scipy ≥ 1.7, Python ≥ 3.9.

## Access tiers

Numerail provides three tiers of access. Each tier adds capability without changing the underlying enforcement logic.

**Tier 1: Pure geometry.** `enforce(vector, region)` with numpy arrays. No schema, no system, no budget, no audit. Just the mathematical guarantee.

```python
import numpy as np
from numerail import enforce, box_constraints, EnforcementResult

region = box_constraints(lower=[0, 0, 0], upper=[1, 1, 1])

result = enforce(np.array([0.5, 0.3, 0.7]), region)
assert result.result == EnforcementResult.APPROVE

result = enforce(np.array([1.5, 0.3, -0.2]), region)
assert result.result == EnforcementResult.PROJECT
assert region.is_feasible(result.enforced_vector)
```

**Tier 2: Named enforcement.** Add `Schema` for dict-to-vector translation with optional normalization, plus `EnforcementConfig` for hard walls, forbidden dimensions, and routing.

```python
from numerail import enforce, box_constraints, Schema, EnforcementConfig

schema = Schema(fields=["amount", "risk_score"], normalizers={"amount": (0, 10000)})
region = box_constraints([0, 0], [0.5, 0.3], names=["max_amt", "max_risk", "min_amt", "min_risk"])
config = EnforcementConfig(hard_wall_constraints=frozenset({"max_risk"}))

vec = schema.vectorize({"amount": 7500, "risk_score": 0.1})
output = enforce(vec, region, config, schema)
```

**Tier 3: Full system.** `NumerailSystem` with budgets, rollback, audit, metrics, trusted context, and region versioning.

```python
from numerail import NumerailSystem, Schema, BudgetSpec, box_constraints, halfplane, combine_regions, EnforcementConfig

schema = Schema(fields=["amount", "risk_score"], normalizers={"amount": (0, 1000)})
region = combine_regions(
    box_constraints([0, 0], [0.5, 0.3], names=["max_amt", "max_risk", "min_amt", "min_risk"]),
    halfplane([1, 0], 0.5, name="daily_cap"),
)
config = EnforcementConfig(mode="hybrid", max_distance=0.5, hard_wall_constraints=frozenset({"max_risk"}))

system = NumerailSystem(schema, region, config)
system.register_budget(BudgetSpec(name="daily", constraint_name="daily_cap", weight={"amount": 1.0}, initial=0.5))
system.set_trusted_fields(frozenset({"risk_score"}))

result = system.enforce({"amount": 200, "risk_score": 0.1}, action_id="txn_001", trusted_context={"risk_score": 0.18})
```

## Constraint types

**LinearConstraints — Ax ≤ b.** The workhorse. Defines a convex polytope via m linear inequalities in n dimensions. Constructor validates dimensions, finiteness, and name uniqueness. The `project_hint` performs iterative half-space projection onto the most-violated row. Key methods: `evaluate(x)`, `is_satisfied(x, tol)`, `project_hint(x)`, `row_violations(x)`, `row_bindings(x, tol)`, `with_bound(row_name, new_bound)`, `with_safety_margin(margin)`.

**QuadraticConstraint — x'Qx + a'x ≤ b.** Ellipsoidal constraint. Q must be positive semidefinite, verified at construction via eigenvalue check. Non-PSD Q raises `ConstraintError`. The `project_hint` uses KKT bisection on the dual variable λ with 100 iterations of binary search.

**SOCPConstraint — ‖Mx + q‖ ≤ c'x + d.** Second-order cone constraint. Models norm bounds, force closure conditions, and robustness margins. No closed-form `project_hint`; Dykstra falls back to single-constraint SLSQP.

**PSDConstraint — A₀ + Σ xᵢAᵢ ≽ 0.** Linear matrix inequality. `evaluate(x)` returns −λ_min(A(x)). Post-check is O(k³) eigendecomposition. No closed-form `project_hint`.

## Builder functions

`box_constraints(lower, upper, names?, tag?)` → FeasibleRegion. Produces 2n linear constraints.

`halfplane(weights, bound, name?, tag?)` → FeasibleRegion. Single half-plane w'x ≤ bound.

`combine_regions(*regions)` → FeasibleRegion. Intersection of all input regions.

`ellipsoid(Sigma, center?, radius_sq?, name?, tag?)` → QuadraticConstraint. Convenience builder for (x−c)'Σ⁻¹(x−c) ≤ r².

## The enforce() function

```python
def enforce(vector, region, config=None, schema=None, prev_hash="") -> EnforcementOutput
```

The eight-step pipeline: (1) post-check input → APPROVE if feasible, (2) hard wall gate → REJECT if hard wall violated, (3) reject mode gate → REJECT if mode is "reject", (4) project via solver chain, (5) post-check solver output → REJECT if fails, (6) dimension policy check → REJECT if forbidden dimension changed, (7) routing threshold check → REJECT if distance exceeds hard_reject, (8) hybrid distance check → REJECT if distance exceeds max_distance.

Steps 1–5 mirror the mathematical proof (PROOF.md). Steps 6–8 can only add rejection, never bypass the post-check.

When a `schema` is provided, `dimension_policies` keys are validated against schema field names. Unknown keys raise `SchemaError` immediately.

## EnforcementConfig

| Field | Default | Description |
|---|---|---|
| `mode` | `"project"` | `"project"`, `"reject"`, or `"hybrid"` |
| `max_distance` | `None` | Required for hybrid mode |
| `dimension_policies` | `{}` | Per-field projection control |
| `routing_thresholds` | `None` | Distance-to-escalation mapping |
| `hard_wall_constraints` | `frozenset()` | Constraint names that trigger immediate reject |
| `safety_margin` | `1.0` | (0, 1] — contracts linear bounds inward |
| `solver_max_iter` | `2000` | SLSQP iteration budget |
| `solver_tol` | `1e-6` | Post-check tolerance |
| `dykstra_max_iter` | `10000` | Dykstra iteration budget |

`safety_margin`: multiplies positive b values by margin (tightening upper bounds), divides negative b values by margin (tightening lower bounds). Both directions contract inward.

## EnforcementOutput fields

| Field | Type | Description |
|---|---|---|
| `result` | `EnforcementResult` | APPROVE, PROJECT, or REJECT |
| `original_vector` | `np.ndarray` | Input vector (copy) |
| `enforced_vector` | `np.ndarray` | Output vector (copy) |
| `distance` | `float` | Euclidean distance, original to enforced |
| `violated_constraints` | `tuple[str]` | Constraints violated by the original |
| `violation_magnitudes` | `dict[str, float]` | Per-constraint violation amounts |
| `binding_constraints` | `tuple[str]` | Constraints tight at the enforced point |
| `solver_method` | `str` | `"none"`, `"box_clamp"`, `"slsqp"`, or `"dykstra"` |
| `iterations` | `int` | Solver iteration count |
| `timestamp` | `str` | UTC ISO 8601 |
| `region_version` | `str` | Version of the active region |
| `routing_decision` | `RoutingDecision` or None | Escalation tier |
| `flagged_dimensions` | `tuple[str]` | Dimensions that triggered PROJECT_WITH_FLAG |
| `prev_hash` | `str` | Audit chain link |

## Solver chain

Three solvers tried in sequence, each independently post-checked:

1. **Box clamp** — O(n), exact for pure-box polytopes.
2. **SLSQP** — General-purpose constrained optimization via scipy.
3. **Dykstra** — Guaranteed convergence for convex intersections, up to 10,000 iterations.

First solver whose output passes `is_feasible()` wins. If all fail, REJECT.

## Hard walls and forbidden dimensions

**Hard walls**: named constraints where violation triggers immediate REJECT, before any solver runs. For categorical violations (sanctioned entity, granting runtime authority when trusted headroom is already exhausted) where projection to the boundary is semantically meaningless.

**Forbidden dimensions**: fields where correction is not permitted. If projection would change a forbidden field, the result is REJECT. The agent must propose the correct value explicitly.

**PROJECT_WITH_FLAG**: correction is allowed but flagged in `output.flagged_dimensions` for downstream attention.

Hard wall names and dimension policy keys are validated at `NumerailSystem` construction time. Unknown names raise errors immediately.

## Routing thresholds

Map correction distance to escalation tiers: SILENT_PROJECT (d ≤ silent), FLAGGED_PROJECT (d ≤ flagged), CONFIRMATION_REQUIRED (d ≤ confirmation), HARD_REJECT (d > hard_reject → becomes REJECT).

## Budgets

Budgets tighten linear constraint bounds as resources are consumed. At enforcement time, remaining = initial − consumed, and the targeted row's bound is replaced with remaining.

Canonical budget grammar (weight map):

```python
"budgets": [{
    "name": "daily_spend",
    "constraint_name": "spend_cap",
    "weight": {"amount": 1.0, "fee": 1.0},
    "initial": 10000.0,
    "mode": "nonnegative"
}]
```

Shorthand for single-field budgets: `"dimension_name": "amount", "weight": 1.0` is accepted as a convenience.

Three consumption modes: `nonnegative` (default, prevents negative-spend exploits), `abs` (accumulates regardless of sign), `raw` (allows negative consumption for backward compat).

Under nonnegative mode, the feasible region is monotone non-expanding: F_{t+1} ⊆ F_t (Theorem 5).

Budget-state region versions encode both the base policy version and a SHA-256 digest of current budget state.

## Rollback

Restores budget state after failed downstream execution. Subtracts the exact stored delta from consumed totals. Returns `RollbackResult(rolled_back=True/False, audit_hash=...)`. `bool(result)` gives `True` if the rollback succeeded. Second rollback of the same action_id raises `ValueError` at the service layer, or returns `RollbackResult(rolled_back=False)` at the engine layer. Immediately syncs the region.

## Audit chain

SHA-256 hash-linked, append-only log. Each record includes `prev_hash` in the hashed payload. `verify()` walks the chain, recomputes every hash, detects any modification. Supports bounded memory via `max_records` with correct verification after eviction.

## Metrics

Thread-safe collection of: approve/project/reject rates, correction distance distribution (mean, p99, max), solver distribution, top violated constraints, top binding constraints.

## Region versioning

Immutable version history. Each state change produces a new FeasibleRegion with a unique version string. Supports rollback to any retained version. Keeps up to 100 versions (configurable).

## Trusted context

`merge_trusted_context(raw_values, trusted_context, trusted_fields)` → merged dict. Only fields in `trusted_fields` are overwritten. Both raw and merged values recorded in feedback for auditability.

`NumerailSystem.set_trusted_fields(fields)` declares trusted fields, validated against schema.

`NumerailSystem.enforce(..., trusted_context={...})` applies merge before vectorization.

## Configuration-driven factory

`NumerailSystem.from_config(config_dict)` constructs a complete system from a JSON-compatible dict. Supports all constraint types, enforcement config, budgets, and trusted fields. Validates dimension policies, hard wall names, budget targets, and trusted field names at construction time. Accepts both `"tags"` and `"authorities"` keys for backward compatibility.

## Analysis utilities

`check_feasibility(region)` → (bool, point or None). Exact for pure-linear (LP); best-effort for mixed (projection search).

`chebyshev_center(region)` → (center or None, radius). Largest inscribed ball for linear regions.

## Feedback synthesis

`synthesize_feedback(output, region, schema, budget_status?)` → structured dict with result, modifications, corrections, distance, binding constraints, budget status, and human-readable message.

## Exception hierarchy

All inherit from `NumerailError`: `ValidationError` (NaN, Inf, dimension mismatch), `ConstraintError` (non-PSD Q, zero weights, shape mismatch, duplicate names), `InfeasibleRegionError` (available for user code), `SolverError` (SLSQP convergence failure), `SchemaError` (missing field, unknown dimension policy key), `ResolutionError` (constraint name not found, hard wall name unknown).

## Early validation (fail-fast)

`dimension_policies` keys validated against schema fields at NumerailSystem construction, enforce() invocation, and from_config(). `hard_wall_constraints` names validated against region at NumerailSystem construction. Budget constraint_names validated as linear rows at register_budget(). All raise immediately on misconfiguration.

## NumerailSystem API

| Method | Signature | Returns |
|---|---|---|
| `enforce` | `(values, action_id?, trusted_context?)` | `Result` |
| `rollback` | `(action_id)` | `bool` |
| `register_budget` | `(spec: BudgetSpec)` | `None` |
| `set_trusted_fields` | `(fields: frozenset)` | `None` |
| `budget_status` | `()` | `dict` |
| `add_constraints` | `(new)` | `FeasibleRegion` |
| `replace_region` | `(new)` | `FeasibleRegion` |
| `rollback_region` | `(version)` | `FeasibleRegion or None` |
| `check_feasibility` | `()` | `(bool, ndarray or None)` |
| `chebyshev_radius` | `()` | `float` |
| `verify_audit` | `()` | `(bool, int)` |
| `export_audit` | `()` | `list[dict]` |
| `get_metrics` | `()` | `dict` |
| `reset_metrics` | `()` | `None` |
| `from_config` | `(config_dict)` | `NumerailSystem` (classmethod) |

## Backward compatibility

`ActionSchema = Schema`, `GCESystem = NumerailSystem`, `GCEError = NumerailError`. The `Polytope` class provides the full v2 API with `.as_region()`, `.contains()`, `.violations()`, `.to_dict()/.from_dict()`, and accepts both `tags=` and `authorities=`.

## Formal properties (complete list)

1. **Soundness (Theorem 1).** APPROVE/PROJECT ⟹ y ∈ F_t.
2. **Fail-Closed (Theorem 2).** All solvers fail ⟹ REJECT.
3. **Hard-Wall Dominance (Theorem 3).** Hard wall violated ⟹ REJECT, no solver invoked.
4. **Forbidden-Dimension Safety (Theorem 4).** Forbidden dimension changed ⟹ REJECT.
5. **Budget Monotonicity (Theorem 5).** Nonneg mode ⟹ F_{t+1} ⊆ F_t.
6. **Rollback Restoration (Theorem 6).** Rollback restores exact delta.
7. **Monotone Self-Limits (Theorem 7).** Reject mode + version control ⟹ no re-arm.
8. **Audit-Chain Integrity (Theorem 8).** SHA-256 collision resistance ⟹ tamper-detectable.
9. **Passthrough / Idempotence.** Feasible input ⟹ APPROVE, distance 0. enforce(enforce(x)) ⟹ APPROVE.

## Certification

Run `pytest tests/test_guarantee.py -v` to independently verify the guarantee. The suite tests 7 categories (structural verification, formal properties, constraint type coverage, adversarial probes, randomized stress tests, mode coverage, tolerance precision).

Run `python proof/verify_proof.py` to validate every axiom, lemma, theorem, and corollary against the engine code (3,732 checks).

Current certification: 45 guarantee tests, 0 failures. 3,732 proof verification checks, 0 failures.

---

*The geometry is certain. The specification is human. The conditions under which they compose are explicit. The combination is the work.*
