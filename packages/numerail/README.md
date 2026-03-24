# Numerail

**Deterministic geometric enforcement for AI actuation safety.**

When an AI proposes a numerical action — a token budget, a GPU lease, an API-call grant, a trade size, a voltage setpoint — Numerail checks it against convex geometric constraints and returns **APPROVE**, **PROJECT** (corrected to the nearest feasible point), or **REJECT**. The solver is untrusted. The guarantee comes from the post-check.

## The Guarantee

```
If enforce() returns APPROVE or PROJECT, the enforced output satisfies
every active constraint to within tolerance τ.

r ∈ {APPROVE, PROJECT} ⟹ ∀ c ∈ F_t.constraints : c.evaluate(y) ≤ τ
```

This holds for all proposed inputs, all constraint type combinations, and all solver implementations. The post-check is the trust boundary. The solver is untrusted. If the engine cannot verify admissibility, it rejects. Nothing reaches the world as "approved" or "projected" without passing the combined feasibility checker.

The guarantee is proved in [`proof/PROOF.md`](proof/PROOF.md) and independently verified by `proof/verify_proof.py` (3,732 checks) and `tests/test_guarantee.py` (45 tests across 7 categories).

## Quickstart

```bash
pip install -e .
```

```python
import numpy as np
import numerail as nm

# Define a feasible region: all values between 0 and 1
region = nm.box_constraints([0, 0], [1, 1])

# An AI proposes a value outside the region
output = nm.enforce(np.array([1.5, 0.5]), region)

print(output.result.value)       # "project"
print(output.enforced_vector)    # [1.0, 0.5]

# THE GUARANTEE
assert region.is_feasible(output.enforced_vector)
```

## Constraint Types

Numerail supports four constraint types in any combination:

- **Linear** (`Ax ≤ b`) — box bounds, halfplanes, polytopes
- **Quadratic** (`x'Qx + a'x ≤ b`) — ellipsoids, energy bounds
- **SOCP** (`‖Mx + q‖ ≤ c'x + d`) — norm bounds, robustness margins
- **PSD** (`A₀ + Σ xᵢAᵢ ≽ 0`) — linear matrix inequalities

## Architecture

```
numerail.engine    — mathematical kernel (enforce, constraints, schema, system)
numerail.parser    — strict policy parser + lint_config
numerail.service   — production runtime service
numerail.local     — in-memory local mode (exercises production code path)
numerail.protocols — typed Protocol interfaces for production repositories
```

The engine is a single file (`src/numerail/engine.py`) with no dependencies beyond numpy and scipy. The production layer wraps it in transactional infrastructure without modifying the enforcement path.

## Verification

```bash
# Run the full test suite (153 tests)
pytest tests/ -v

# Run only the guarantee certification suite (45 tests, 7 categories)
pytest tests/test_guarantee.py -v

# Run the machine-verifiable proof checker (3,732 checks)
cd proof && python verify_proof.py
```

## Examples

[`examples/ai_resource_governor.py`](examples/ai_resource_governor.py) — Base AI governance: token ceilings, GPU leases, API-call grants, concurrency limits, trusted infrastructure telemetry, and cumulative daily budgets — with all four constraint types, trusted-context injection, budget depletion, rollback, and audit.

[`examples/ai_circuit_breaker.py`](examples/ai_circuit_breaker.py) — Advanced control-plane reserve pattern: extends the resource governor with trusted controller reserves and disturbance margins so that any approved action provably preserves enough capacity for the governance system itself to keep running.

## Documentation

- [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) — **Start here.** Complete walkthrough, quick-start, and API guide
- [`docs/GUARANTEE.md`](docs/GUARANTEE.md) — What the guarantee is, why it holds, what it depends on
- [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) — How to write constraint specifications
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Production deployment guide
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — API reference, config grammar, constraint types
- [`proof/PROOF.md`](proof/PROOF.md) — Mathematical proof (Axiom 1, Lemmas 1–3, Theorems 1–9)

## License

MIT. See [LICENSE](LICENSE).
