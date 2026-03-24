# Numerail

[![CI](https://github.com/Numerail/Numerail/actions/workflows/ci.yml/badge.svg)](https://github.com/Numerail/Numerail/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Deterministic geometric enforcement for AI actuation safety.**

Silicon Valley shipped an incomplete product — an unverified proposal system
(LLM) — and we have begun to automate critical real-world tasks with it. This
will not scale. We cannot take probabilistic AI at its word because there will
always be a nonzero chance of hallucination or misalignment, so we must reduce
every consequential AI decision to a number, and verify every number before it
touches the world.

This is the insight: AI does not act in words. It acts in numbers. Token
counts, resource budgets, voltage setpoints, trade sizes, lease durations.
Every real-world consequence of an AI system passes through a numerical
bottleneck before it becomes reality via tool calls, APIs and MCPs. If you can
define the geometry of what is permissible, you can enforce it with
mathematical certainty. Not by trusting the model, but by checking the output
against policy before it becomes an irreversible action.

Numerail makes this possible. It places a deterministic enforcement boundary
between what the AI proposes and what the world receives, guaranteeing that
every approved action satisfies every active constraint with 100% policy
compliance. The model remains free to be creative, exploratory, and
probabilistic within the bounds; it simply cannot exceed them. This means the
hard problem is no longer "can we trust the AI?" It is "can we write the right
constraints?"

Specification, not total alignment, becomes the bottleneck. And specification
is a human-legible, auditable, improvable engineering problem.

---

When an AI proposes a numerical action — a token budget, a GPU lease, an
API-call grant, a trade size, a voltage setpoint — Numerail checks it against
convex geometric constraints and returns **APPROVE**, **PROJECT** (corrected to
the nearest feasible point), or **REJECT**. The solver is untrusted. The
guarantee comes from the post-check.

---

## The Guarantee

```
If enforce() returns APPROVE or PROJECT, the enforced output satisfies
every active constraint to within tolerance τ.

  r ∈ {APPROVE, PROJECT}  ⟹  ∀ c ∈ F_t : c.evaluate(y) ≤ τ
```

This holds for all proposed inputs, all constraint combinations, and all solver
implementations. The guarantee is proved in
[`packages/numerail/proof/PROOF.md`](packages/numerail/proof/PROOF.md) and
independently verified by 3,732 machine-checked assertions and 45 certification
tests.

---

## Packages

This repository contains two packages with a clean dependency relationship.

### `numerail` — core enforcement kernel

The mathematical kernel. Zero external dependencies beyond numpy and scipy.

```
packages/numerail/
```

| What | Value |
|---|---|
| Version | 5.0.0 |
| Python | ≥ 3.9 |
| Dependencies | numpy ≥ 1.21, scipy ≥ 1.7 |
| License | MIT |
| Tests | 153 passing |
| Proof checks | 3,732 passing |

Provides the enforcement guarantee, four convex constraint types (linear,
quadratic, SOCP, PSD), the solver chain, schema, budgets, audit chain, and the
full production service layer.

→ [Core README](packages/numerail/README.md)

### `numerail-ext` — survivability extension

Supervisory degradation around the core kernel. Requires `numerail`.

```
packages/numerail_ext/
```

| What | Value |
|---|---|
| Version | 0.4.0 |
| Python | ≥ 3.10 |
| Dependencies | numerail ≥ 5.0.0, numpy ≥ 1.21, scipy ≥ 1.7 |
| License | MIT |
| Tests | 207 passing |

Provides the breaker state machine, `IncidentCommanderTransitionModel`,
`StateTransitionGovernor`, global default policy pack, and
`NumerailPolicyContract` (content-addressable, chain-linked policy interchange).

→ [Extension README](packages/numerail_ext/README.md)

---

## Quickstart

**Install core:**

```bash
cd packages/numerail
pip install -e .
```

```python
import numpy as np
import numerail as nm

region = nm.box_constraints([0, 0], [1, 1])
output = nm.enforce(np.array([1.5, 0.5]), region)

print(output.result.value)       # "project"
print(output.enforced_vector)    # [1.0, 0.5]

assert region.is_feasible(output.enforced_vector)  # THE GUARANTEE
```

**Install the survivability extension:**

```bash
cd packages/numerail_ext
pip install -e .
```

---

## Verification

```bash
# Core — full test suite (153 tests)
cd packages/numerail && pytest tests/ -v

# Core — guarantee certification only (45 tests)
cd packages/numerail && pytest tests/test_guarantee.py -v

# Core — machine-verifiable proof checker (3,732 checks)
cd packages/numerail && python proof/verify_proof.py

# Ext — full test suite (207 tests)
cd packages/numerail_ext && pytest tests/ -v
```

---

## Repository Layout

```
numerail/                        ← repository root
  packages/
    numerail/                    ← core enforcement kernel
      src/numerail/
        engine.py                ← mathematical kernel (single file)
        parser.py                ← policy parser + lint_config
        service.py               ← production runtime service
        local.py                 ← in-memory local mode
        protocols.py             ← typed Protocol interfaces
        errors.py                ← production-layer exceptions
      tests/                     ← 153 tests
      proof/                     ← PROOF.md + verify_proof.py (3,732 checks)
      docs/                      ← DEVELOPER_GUIDE, GUARANTEE, SPECIFICATION,
      examples/                  ←   DEPLOYMENT, REFERENCE
    numerail_ext/                ← survivability extension
      src/numerail_ext/survivability/
        breaker.py               ← BreakerStateMachine
        transition_model.py      ← IncidentCommanderTransitionModel
        policy_builder.py        ← build_v5_policy_from_envelope()
        global_default.py        ← build_global_default() — policy pack
        governor.py              ← StateTransitionGovernor
        contract.py              ← NumerailPolicyContract
        local_backend.py         ← LocalNumerailBackend
        validation.py            ← validate_receipt_against_grant()
        types.py                 ← shared data types and Protocols
      tests/                     ← 207 tests
  .github/workflows/             ← CI
  .gitignore
  CHANGELOG.md
  README.md                      ← this file
```

---

## Design

Twelve engineering principles govern every component. The most foundational:

**The guarantee is the product.** Everything else exists to make Theorem 1
useful, composable, and auditable. No feature is added that weakens it.

**The AI is the subject of governance, never the governor.** The AI proposes
13 numerical values. Everything else — the 17 trusted fields, the breaker mode,
the envelope ceilings, the budget remaining, the freshness window — is
determined by the orchestrator from server-authoritative sources the AI cannot
read or write.

**The kernel is unchanged by the layers above it.** The breaker suite controls
which policy the kernel enforces against. It cannot modify how enforcement
works. The guarantee proved for the kernel propagates to every layer built on
top of it.

**The default is denial.** Every permission must be explicitly granted through
constraint geometry. The system does not guess intent.

---

## Documentation

| Document | Location |
|---|---|
| Developer guide (start here) | [`packages/numerail/docs/DEVELOPER_GUIDE.md`](packages/numerail/docs/DEVELOPER_GUIDE.md) |
| The guarantee | [`packages/numerail/docs/GUARANTEE.md`](packages/numerail/docs/GUARANTEE.md) |
| The specification problem | [`packages/numerail/docs/SPECIFICATION.md`](packages/numerail/docs/SPECIFICATION.md) |
| Deployment guide | [`packages/numerail/docs/DEPLOYMENT.md`](packages/numerail/docs/DEPLOYMENT.md) |
| API reference | [`packages/numerail/docs/REFERENCE.md`](packages/numerail/docs/REFERENCE.md) |
| Mathematical proof | [`packages/numerail/proof/PROOF.md`](packages/numerail/proof/PROOF.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |

---

## License

MIT — see [`packages/numerail/LICENSE`](packages/numerail/LICENSE).
