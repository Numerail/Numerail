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
independently verified by 3,732 machine-checked assertions, 46 certification
tests, and a 99-test mathematical guarantee analysis suite that independently
verifies every theorem against the live codebase.

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
| Tests | 253 passing |
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
| Tests | 217 passing |

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
# Core — full test suite (253 tests)
cd packages/numerail && pytest tests/ -v

# Core — guarantee certification only (46 tests)
cd packages/numerail && pytest tests/test_guarantee.py -v

# Core — mathematical guarantee analysis (99 tests, one per proof claim)
cd packages/numerail && pytest tests/test_mathematical_guarantees.py -v

# Core — machine-verifiable proof checker (3,732 checks)
cd packages/numerail && python proof/verify_proof.py

# Core — Rocq machine-checked proof (11 theorems, 0 Admitted) — requires Rocq 9.0 / Coq 8.18+
cd packages/numerail/proof && coqc Guarantee.v

# Core — Lean 4 machine-checked proof (12 theorems, 0 sorry) — requires Lean 4 + Mathlib
cd packages/numerail/proof && lake env lean Guarantee.lean

# Ext — full test suite (207 tests)
cd packages/numerail_ext && pytest tests/ -v

# Ext — integration tests only (10 tests, exercises full governor lifecycle)
cd packages/numerail_ext && pytest tests/test_integration.py -v

# Full stack — hello world smoke test (14 steps, all theorems exercised)
python packages/numerail/examples/hello_world.py
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
      tests/                     ← 253 tests
        test_guarantee.py        ← 46 certification tests (proof/PROOF.md §Theorem 1–9)
        test_mathematical_guarantees.py ← 99 guarantee analysis tests (one per proof claim)
      proof/                     ← PROOF.md + verify_proof.py (3,732 checks) + Guarantee.v + Guarantee.lean
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
      tests/                     ← 217 tests
        test_integration.py      ← 10 integration tests (full governor lifecycle)
  .github/workflows/             ← CI
  .gitignore
  CHANGELOG.md
  README.md                      ← this file
```

---

## Design

Numerail implements a propose-check-enforce architecture. The AI proposes. The
geometry checks. The engine enforces. Nothing reaches the world unchecked.

This is the same design primitive that governs fly-by-wire flight. When a pilot
— or autopilot — commands a maneuver, the flight control computer does not pass
the command to the control surfaces. It checks the command against the
aerodynamic envelope: angle of attack limits, load factor boundaries, stall
margins. If the command is inside the envelope, it executes. If it is outside,
the computer corrects it to the nearest safe equivalent or rejects it. The
pilot retains full authority within the envelope but cannot exceed it. This is
why fly-by-wire aircraft do not depart controlled flight — not because pilots
are perfect, but because the envelope is enforced geometrically, after the
proposal and before the actuator. The same principle governs robotic workspace
safety, where a safety controller checks proposed trajectories against convex
collision boundaries before the motors execute.

Numerail applies this architecture to AI actuation. The AI is the pilot — it
proposes a workload vector. The feasible region is the envelope — convex
constraints encoding what is operationally permissible. The enforcement engine
is the flight control computer — it approves, projects to the nearest feasible
point, or rejects. Everything else in the system exists to make this loop
auditable, adaptive, and survivable.

Four principles govern the design:

**The guarantee is the product.** Theorem 1 proves that if the engine returns
APPROVE or PROJECT, the output satisfies every active constraint. The service
layer, the budgets, the audit chain, the breaker suite — all exist to make this
guarantee useful in production. No feature is added that weakens it.

**The AI is the subject of governance, never the governor.** The AI proposes
workload values. Everything else — live telemetry, controller reserves, breaker
mode, budget remaining, freshness windows — is determined by the orchestrator
from server-authoritative sources the AI cannot read or write.

**The kernel is unchanged by the layers above it.** The survivability extension
controls which policy the kernel enforces against. It cannot modify how
enforcement works. The guarantee proved for the kernel propagates unchanged to
every layer built on top of it.

**The default is denial.** Every permission must be explicitly granted through
constraint geometry. An empty policy rejects everything. The system is safe by
construction, not by assumption.

---

## Challenges

Numerail guarantees that every approved or projected output satisfies every
active constraint. It does not guarantee that the constraints are correct. The
gap between these two statements is where every real deployment challenge lives.

**The specification problem is the hard problem.** Writing the right constraints
— translating governance intent into convex geometry that captures all relevant
risk, at the right levels, with the right interactions, for the right fields —
is the hardest part of deploying Numerail. A credit cap set at $5,000 when the
intent was $50 will be faithfully enforced at $5,000. A policy that omits a
safety-relevant dimension leaves that dimension unconstrained. A joint constraint
that weights two fields equally when the domain requires asymmetric weighting
will enforce the wrong tradeoff with perfect fidelity. The engine cannot detect
these errors because they are not errors in the engine — they are errors in the
specification.

**Projection changes the AI's proposal, and the change may not match intent.**
When the engine projects an infeasible vector to the nearest feasible point, the
result satisfies every constraint — that is proved. But "nearest feasible point"
is a geometric operation, not a semantic one. If an AI proposes 200 GPU-seconds
and the cap is 120, projection clips it to 120. That is probably what the
deployer intended. But if an AI proposes a workload that violates a complex joint
constraint involving GPU utilization, API calls, and parallelism simultaneously,
the projected point will be the closest point in the feasible polytope — which
may redistribute the workload across dimensions in ways the deployer did not
anticipate. The projected action is safe by the constraint definition. Whether it
is operationally sensible depends on whether the constraints correctly encode the
deployer's notion of sensible. For dimensions where silent correction is
dangerous, the `PROJECTION_FORBIDDEN` dimension policy exists — it forces
rejection instead of projection, requiring the AI to propose a feasible action on
its own.

**Breaker mode transitions depend on threshold calibration.** The breaker state
machine transitions between CLOSED, THROTTLED, HALF_OPEN, SAFE_STOP, and OPEN
based on a weighted overload score computed from live telemetry. The weights
(0.30 GPU, 0.25 API, 0.20 DB, 0.10 queue, 0.15 error rate) and the threshold
values (trip_score, reset_score, safe_stop_score) determine when the system
degrades. If the trip threshold is too low, the system throttles under normal
load and rejects actions that should be permitted. If it is too high, the system
stays in CLOSED mode under genuine stress and permits actions that should be
restricted. If the reset threshold is too close to the trip threshold, the
breaker oscillates between modes. If the safe_stop threshold is too low,
SAFE_STOP latches prematurely and requires manual intervention when automated
recovery would have been appropriate. None of these are bugs in the breaker logic
— the transitions are correct for whatever thresholds you configure. They are
calibration problems, and they require operational experience with the specific
infrastructure being governed.

**Budget sizing determines how long authority lasts.** The three shift budgets
(GPU-seconds, external API calls, cloud mutations) deplete monotonically as the
AI consumes resources. If the GPU budget is set to 3,600 seconds for an 8-hour
shift, the AI can lease 450 seconds per hour on average. If the AI's workload is
bursty — consuming 2,000 seconds in the first hour — it will hit budget
exhaustion well before the shift ends, and every subsequent action that requires
GPU time will be rejected. The budget is not wrong; the budget is exactly what
was specified. But the deployer may have intended a burst-tolerant budget with
per-hour sub-limits rather than a single-shift pool. Numerail provides the budget
tracking and enforcement machinery. The deployer must decide the budget
structure.

**Reserve sizing determines whether the governance system survives.** The
control-plane reserve pattern subtracts protected capacity for the governance
controller before evaluating the AI's request. If the GPU reserve is set to 30
seconds but the controller actually requires 50 seconds to complete its
monitoring and enforcement cycle under load, the reserve is insufficient and the
controller can be starved even though the constraints say it should be protected.
The constraint will be satisfied — the 30-second reserve will be preserved. But
30 seconds is not enough. This is a specification error in the reserve value, not
a failure of the enforcement engine.

**Trusted context accuracy is assumed, not verified.** The server-authoritative
fields — current GPU utilization, API utilization, error rates, controller
reserves, disturbance margins — are injected by the orchestrator from
infrastructure telemetry. If the telemetry source reports GPU utilization as 0.30
when the actual utilization is 0.85, the enforcement engine will evaluate the
AI's request against a constraint that thinks there is 55% more headroom than
actually exists. The engine guarantees that the output satisfies every constraint
as computed from the injected values. It does not guarantee that the injected
values are accurate. Stale telemetry is partially mitigated by the three-layer
freshness architecture (engine constraints, governor clock check, reservation
expiry), but telemetry that is fresh and wrong — a sensor reporting an incorrect
value — passes through the freshness checks and corrupts the constraint
evaluation silently.

None of this is unique to Numerail. Every rule-based governance system in every
regulated industry faces the same problem: building codes, aviation maintenance procedures,
flight envelopes, financial compliance rules. The constraints are written by
humans, and humans can write the wrong constraints. Numerail's contribution is to
make the enforcement provably correct so that specification becomes the only
remaining problem — and specification is a legible, auditable, iterative
engineering problem with established tools, observable metrics, and a versioned
policy history that supports continuous refinement.

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
