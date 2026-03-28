# Numerail

[![CI](https://github.com/Numerail/Numerail/actions/workflows/ci.yml/badge.svg)](https://github.com/Numerail/Numerail/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Numerail/Numerail/blob/main/LICENSE)

**Deterministic geometric enforcement for AI actuation safety.**

Silicon Valley shipped a beautiful but incomplete product — an unverified proposal system (LLM) — and we have begun to automate critical real-world tasks with it. This will not scale. We cannot take probabilistic AI at its word because there will always be a nonzero chance of hallucination or misalignment, so we must reduce every consequential AI decision to a number, and verify every number before it touches the world.

This is the core insight: AI does not act in words. It acts in numbers. Token counts, resource budgets, voltage setpoints, trade sizes, lease durations. Every real-world consequence of an AI system passes through a numerical bottleneck before it becomes reality via tool calls, APIs and MCPs. If you can define the geometry of what is permissible, you can enforce it with mathematical certainty. Not by trusting the model, but by checking the output against policy before it becomes an irreversible action.

Numerail makes this possible. It places a deterministic enforcement boundary between what the AI proposes and what the world receives, guaranteeing that every approved action satisfies every active constraint with 100% policy compliance. The model remains free to be creative, exploratory, and probabilistic within the bounds; it simply cannot exceed them. This means the hard problem is no longer "can we trust the AI?" It is "can we write the right constraints?"

Specification, not total alignment, becomes the bottleneck. And specification is a human-legible, auditable, improvable engineering problem.

When an AI proposes a numerical action — a token budget, a GPU lease, an API-call grant, a trade size, a voltage setpoint — Numerail checks it against convex geometric constraints and returns **APPROVE**, **PROJECT** (corrected to the nearest feasible point), or **REJECT**. The solver is untrusted. The guarantee comes from the post-check.

---

## The Guarantee
If enforce() returns APPROVE or PROJECT, the enforced output satisfies
every active constraint to within tolerance τ.
r ∈ {APPROVE, PROJECT}  ⟹  ∀ c ∈ F_t : c.evaluate(y) ≤ τ

This holds for all proposed inputs, all constraint combinations, and all solver implementations. The guarantee is proved in [`packages/numerail/proof/PROOF.md`](packages/numerail/proof/PROOF.md) and independently verified by 3,732 machine-checked assertions, 46 certification tests, and a 99-test mathematical guarantee analysis suite that independently verifies every theorem against the live codebase.

---

## What Numerail Provides

### Enforcement kernel

Four families of convex constraints — linear (`Ax ≤ b`), quadratic (`x'Qx + a'x ≤ b`), SOCP (`‖Mx+q‖ ≤ c'x+d`), and PSD (`A₀ + Σ xᵢAᵢ ≽ 0`) — evaluated against every proposed action. The solver chain (box clamping → SLSQP → Dykstra alternating projection) is untrusted by design. Every solver candidate is post-checked by `is_feasible()` before emission. If all solvers fail, the result is REJECT. The APPROVE path takes 25 microseconds. The system evaluates 40,000 enforcement decisions per second on a single core.

### Schema and named dimensions

Human-readable field names mapped to vector indices so constraints can be authored and audited in domain language. Vectorization and devectorization convert between named dictionaries and dense numerical vectors transparently.

### Enforcement modes and dimension policies

Three enforcement modes control what happens when the input is infeasible. **Project** corrects to the nearest feasible point. **Reject** denies without correction. **Hybrid** corrects within a configurable distance threshold and rejects beyond it. Per-dimension policies provide fine-grained control: `PROJECTION_FORBIDDEN` forces rejection instead of silent correction for dimensions where the deployer requires the AI to propose a feasible value on its own.

Four-tier routing classifies every enforcement decision by the severity of the deviation: `SILENT_PROJECT` (small correction, applied silently), `FLAGGED_PROJECT` (moderate correction, flagged for review), `CONFIRMATION_REQUIRED` (large correction, requires human approval), and `HARD_REJECT` (extreme violation, rejected immediately).

### Budget tracking with transactional rollback

Monotonically depleting resource pools with configurable weight maps. Each enforcement action consumes budget. When budget is exhausted, the constraint geometry tightens and further actions in that dimension are rejected. Rollback restores budget by exactly the consumed amount for a specific action, enabling transactional semantics where failed or reversed actions do not permanently consume budget.

### Audit chain

SHA-256 hash-linked, tamper-evident, append-only log recording every enforcement decision, every budget consumption, and every rollback. Each record contains the previous record's hash within its own hashed payload. Any modification, insertion, or deletion of a record breaks the hash chain and is detectable by verification.

### Trusted context injection

The `TrustedContextProvider` protocol ensures server-authoritative fields — current time, infrastructure telemetry, budget state, breaker mode — are injected by the orchestrator, not proposed by the AI. The AI cannot manipulate its own evaluation context. `DefaultTimeProvider` ships out of the box using millisecond precision to preserve float64 exact integer range. Any field the AI proposes for a trusted dimension is overwritten with the authoritative value before enforcement. Mismatches are logged and recorded in the enforcement metadata for forensic analysis.

---

## Survivability Extension

The extension adds operational resilience around the core kernel. The kernel is unchanged by the layers above it — the guarantee proved for the kernel propagates to every layer built on top of it.

### Breaker state machine

Five operational modes with hysteretic transitions: **CLOSED** (full authority), **THROTTLED** (reduced authority), **HALF_OPEN** (probing recovery), **SAFE_STOP** (minimal authority, latched until manual human reset), and **OPEN** (enforcement disabled for emergency bypass). The trip threshold and reset threshold are different, preventing oscillation under fluctuating load. The breaker trips automatically when a weighted overload score — computed from GPU utilization (0.30), API utilization (0.25), DB utilization (0.20), queue depth (0.10), and error rate (0.15) — crosses the configured threshold.

### Monotone authority reduction

Authority caps are monotonically non-increasing through the mode hierarchy. Every workload field's cap in THROTTLED is less than or equal to its cap in CLOSED, and so on down to SAFE_STOP. Degradation always reduces the AI's authority, never increases it. The transition model computes per-mode caps from live infrastructure telemetry across 13 workload fields.

### Governor lifecycle

The `StateTransitionGovernor` orchestrates a 12-step cycle on every enforcement action: read live telemetry, compute overload score, update breaker mode, synthesize authority envelope from current mode and telemetry, compile constraint geometry from the envelope, set the active policy on the enforcement backend, acquire a reservation, enforce the AI's proposal against the compiled constraints, compute the payload digest, issue an executable grant, validate the execution receipt, and verify next-state safety. The constraint geometry adapts in real time to infrastructure conditions.

### Global default policy

Ships locked-down-by-default with 30 schema fields, approximately 80 linear constraints, 1 quadratic energy bound, 1 SOCP burst envelope, 1 PSD coupled-channel constraint, 17 trusted fields marked FORBIDDEN, 3 shift budgets (GPU-seconds, external API calls, cloud mutations), and four-tier escalation routing. Constraint modules reference EU AI Act Articles 9, 12, 14, and 15. An empty policy rejects everything — the system is safe by construction, not by assumption.

### Policy contract system

Content-addressable, SHA-256 digested, chain-linked policy interchange. Every policy version has a unique digest computed from its canonical serialization. Contracts are chained: each references the digest of its predecessor. The complete policy history is tamper-evident and auditable. Any third party can independently verify that the policy in force at time T was exactly the policy that was authorized. Round-trip serialization (to_dict → from_dict, to_json → from_json, to_bytes → from_bytes) preserves digest stability.

---

## Human-in-the-Loop Supervision

The `SupervisedGovernor` wraps the `StateTransitionGovernor` without modifying it. When configurable triggers fire, execution pauses and the action is held for human review.

### Three review profiles

**ADVISORY** — lightest touch. Blocking review only on SAFE_STOP entry and audit chain failure. Asynchronous notification on budget exhaustion. Suitable for low-stakes deployments with well-calibrated constraints.

**SUPERVISORY** — the recommended default. Blocking review on REJECT, CONFIRMATION_REQUIRED projections, PROJECTION_FORBIDDEN rejections, SAFE_STOP entry, budget exhaustion, and audit chain failure. Asynchronous notification on breaker mode changes and flagged projections. Suitable for most production deployments.

**MANDATORY** — full human oversight. Blocking review on every enforcement result that is not a clean APPROVE. Blocking on every breaker transition, every budget event, every policy change. Suitable for high-stakes or early-deployment scenarios.

### Five decision types

**Approve** — the reviewer authorizes execution. The action is re-enforced against the current constraint geometry before it executes (TOCTOU protection). **Deny** — the reviewer blocks execution, with optional guidance bounds for the agent's next attempt. **Modify** — the reviewer proposes a different vector, which is re-enforced before execution in a single pass (no review loop). **Escalate** — the reviewer forwards the action to a higher-authority reviewer, subject to a configurable escalation ceiling. **Defer** — the reviewer extends the timeout once.

### Safety properties

Every action is re-enforced against the current constraint geometry immediately before execution, even after human approval. This closes the time-of-check-to-time-of-execute gap — the guarantee holds at the moment the action reaches the world, not the moment it was originally evaluated. Expired reviews default to denial. Unauthenticated decisions are rejected. Escalation has a ceiling. Defer is permitted once per review. Every human decision is recorded in a SHA-256 hash-linked audit chain alongside the enforcement decisions. Humans cannot bypass the enforcement boundary — they can only approve or deny actions that have already been checked by the engine.

---

## Formal Verification

The enforcement guarantee is not a claim in a whitepaper. It is a mathematical proof, independently machine-checked by two proof assistants built on different kernels with no shared code.

**Rocq (OCaml kernel):** 11 theorems, 0 Admitted. Compiles on Rocq 9.0 / Coq 8.18+. Imports only `Stdlib.List` and `Stdlib.Bool`. Three axioms: abstract vector type, opaque solver with post-check guarantee, abstract operational filters.

**Lean 4 (C++ kernel):** 12 theorems, 0 sorry. Requires Lean 4 with Mathlib. Same three axioms, same theorem statements, independent formalization.

Both prove the central guarantee (enforcement soundness), fail-closed rejection, hard-wall dominance, budget monotonicity, passthrough, and idempotence. The same theorem — that every APPROVE or PROJECT output satisfies every active constraint — is accepted by both kernel implementations.

**Python proof checker:** 3,732 structural and property checks confirming the implementation matches the formal model. Covers all return points, guard conditions, function signatures, constraint evaluation formulas, and randomized property checks.

**Test suite:** 46 certification tests (one per theorem with structural, adversarial, and stress coverage), 99 mathematical guarantee analysis tests (one per proof claim), and 627 total tests across all three packages exercising every code path.

This level of formal verification is, to the best of the author's knowledge, unique among open-source AI safety systems.

---

## Performance

| Metric | Value |
|---|---|
| APPROVE path (median) | 25 µs |
| APPROVE throughput | 40,000 ops/sec (single core) |
| PROJECT path (linear, 8-dim) | 255 µs |
| PROJECT path (quadratic, 8-dim) | 6,072 µs |
| REJECT (mode=reject) | 20 µs |
| Governor full cycle | 133 ms (policy rebuild every cycle) |
| BudgetTracker.rollback | 0.45 µs |
| AuditChain.verify (per record) | 6.5 µs |

The APPROVE path is sub-50 microseconds at every complexity level tested (2-dim through 30-dim). Inline synchronous enforcement adds negligible latency to any AI action — less than a single network round-trip. The system can check every action, on every cycle, on standard hardware, without the AI agent or the end user noticing it is there.

Full benchmark report (72 benchmarks): [`packages/numerail/tests/BENCHMARK_REPORT.md`](packages/numerail/tests/BENCHMARK_REPORT.md)

---

## Packages

This repository contains three packages with a clean dependency relationship.

### `numerail` — core enforcement kernel

The mathematical kernel. Zero external dependencies beyond numpy and scipy.
packages/numerail/

| What | Value |
|---|---|
| Version | 5.0.0 |
| Python | ≥ 3.9 |
| Dependencies | numpy ≥ 1.21, scipy ≥ 1.7 |
| License | MIT |
| Tests | 265 passing |
| Proof checks | 3,732 passing |

Provides the enforcement guarantee, four convex constraint types (linear, quadratic, SOCP, PSD), the solver chain, schema, budgets, audit chain, trusted context injection, and the full production service layer.

→ [Core README](packages/numerail/README.md)

### `numerail-ext` — survivability extension

Supervisory degradation, behavioral circuit breaking, human-in-the-loop supervision, and cryptographic policy provenance around the core kernel. Requires `numerail`.
packages/numerail_ext/

| What | Value |
|---|---|
| Version | 0.4.0 |
| Python | ≥ 3.10 |
| Dependencies | numerail ≥ 5.0.0, numpy ≥ 1.21, scipy ≥ 1.7 |
| License | MIT |
| Tests | 306 passing |

Provides the breaker state machine, `IncidentCommanderTransitionModel`, `StateTransitionGovernor`, global default policy pack, `NumerailPolicyContract` (content-addressable, chain-linked policy interchange), `SupervisedGovernor` (human-in-the-loop enforcement), and `LocalApprovalGateway` for development and testing.

→ [Extension README](packages/numerail_ext/README.md)

### `numerail-learn` — reinforcement learning from enforcement

Converts enforcement decisions into training data for LLMs. Requires `numerail` and `numerail-ext`.
packages/numerail_learn/

| What | Value |
|---|---|
| Version | 0.1.0 |
| Python | ≥ 3.10 |
| Dependencies | numerail ≥ 5.0.0, numerail-ext ≥ 0.4.0, numpy ≥ 1.21 |
| License | MIT |
| Tests | 56 passing |

Provides the enforcement experience buffer, reward shaping (conservative/permissive/strict presets), training data adapters (SFT, DPO, PPO), and the orchestrator for tracking model improvement over time.

**Boundary-seeking mitigation** — training on projected corrections without adjustment teaches the model to operate at the edge of its authority with zero margin. Two mechanisms address this: SFT retraction pulls each supervision target away from the constraint boundary toward the feasible interior (guaranteed feasible by convexity), and an optional margin bonus rewards proposals that maintain distance from the boundary. The tradeoff between authority utilization and safety margin is controlled by the deployer via `retraction_factor` and `margin_bonus_scale`.

→ [Learn README](packages/numerail_learn/README.md)

---

## Reinforcement Learning from Enforcement (Conceptual)

> **Status: Conceptual infrastructure. The experience buffer, reward shaping, and training adapters are implemented and tested (56 tests passing). The training loop itself — connecting enforcement feedback to actual LLM weight updates — has not been tested against a live model. The components below provide the data pipeline. The model training integration is left to the deployer.**

Every Numerail enforcement decision contains a complete training signal. When the engine returns APPROVE, the model proposed something feasible — positive reinforcement. When it returns PROJECT, the engine corrected the proposal to the nearest feasible point — a supervised learning target showing exactly what the model should have proposed. When it returns REJECT, the proposal was infeasible and could not be corrected — negative reinforcement.

This is a stronger alignment signal than standard RLHF for three reasons. The reward is deterministic (the vector satisfies the constraints or it does not — no labeler disagreement). The correction is constructive (PROJECT tells the model the right answer, not just that it was wrong). And the reward is grounded in the actual constraint geometry that will be enforced at inference time (no reward model to overfit).

`numerail-learn` provides the infrastructure to collect and shape these signals:

**Experience buffer** — captures the full context of every enforcement decision: conversation history, tool call, proposed vector, enforcement result, corrected vector, constraint violations, breaker mode, budget state, and computed reward. Thread-safe, fixed-size circular buffer with JSON export/import.

**Reward shaping** — computes training rewards from enforcement outputs with three presets. *Conservative* (high penalty for violations, trains models to stay well within bounds), *permissive* (small penalties, trains models to use available authority efficiently), and *strict* (maximum penalty for any non-APPROVE). Per-dimension feedback identifies which fields the model consistently misjudges.

**Training adapters** — convert enforcement experiences into the specific formats required by LLM training frameworks. *SFT* (supervised fine-tuning on PROJECT corrections — the model learns "you proposed X, you should have proposed Y"), *DPO* (direct preference optimization on APPROVE vs REJECT pairs — the model learns which outputs are preferred), and *PPO* (proximal policy optimization with shaped rewards — compatible with TRL).

**Orchestrator** — coordinates the collect-and-export cycle with approval rate tracking, dimension violation analysis, and improvement reporting over training rounds.

**Boundary-seeking mitigation** — training on projected corrections without adjustment teaches the model to operate at the edge of its authority with zero margin. Two mechanisms address this: SFT retraction pulls each supervision target away from the constraint boundary toward the feasible interior (guaranteed feasible by convexity), and an optional margin bonus rewards proposals that maintain distance from the boundary. The tradeoff between authority utilization and safety margin is controlled by the deployer via `retraction_factor` and `margin_bonus_scale`.

The key metric is **approval rate over time**. A model that starts at 30% APPROVE (most proposals need correction) and reaches 90% after training has internalized the constraint geometry. The enforcement boundary remains in place regardless — training makes the model better at proposing feasible actions, enforcement ensures the boundary holds even when training fails. They are complementary, not redundant.

This is the relationship between control and alignment: the enforcement guarantee is the floor, alignment is the optimization above it. The same system that prevents harm (enforcement) also generates the signal that reduces the probability of harm (training feedback). Neither replaces the other.

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
# Core — full test suite (265 tests)
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

# Ext — full test suite (306 tests)
cd packages/numerail_ext && pytest tests/ -v

# Ext — integration tests only (10 tests, exercises full governor lifecycle)
cd packages/numerail_ext && pytest tests/test_integration.py -v

# Full stack — hello world smoke test (14 steps, all theorems exercised)
python packages/numerail/examples/hello_world.py

# Learn — full test suite (56 tests)
cd packages/numerail_learn && pytest tests/ -v
```

---

## Repository Layout

```
numerail/                        ← repository root
packages/
  numerail/                    ← core enforcement kernel
    src/numerail/
      engine.py                ← mathematical kernel (single file, ~2,400 lines)
      parser.py                ← policy parser + lint_config
      service.py               ← production runtime service
      local.py                 ← NumerailSystemLocal + DefaultTimeProvider
      protocols.py             ← typed Protocol interfaces + HITL types
      errors.py                ← production-layer exceptions
      py.typed                 ← PEP 561 marker
    tests/                     ← 265 tests
      test_guarantee.py        ← 46 certification tests (proof/PROOF.md §Theorem 1–9)
      test_mathematical_guarantees.py ← 99 guarantee analysis tests (one per proof claim)
      test_trusted_context.py  ← trusted context injection tests
      BENCHMARK_REPORT.md      ← 72 performance benchmarks
    proof/
      PROOF.md                 ← Axiom 1, Lemmas 1–3, Theorems 1–9, 2 Corollaries
      verify_proof.py          ← 3,732 structural/property checks
      Guarantee.v              ← Rocq/Coq formalization (11 theorems, 0 Admitted)
      Guarantee.lean           ← Lean 4 + Mathlib formalization (12 theorems, 0 sorry)
    docs/                      ← DEVELOPER_GUIDE, GUARANTEE, SPECIFICATION,
                                  DEPLOYMENT, REFERENCE, REGULATORY
    examples/
      hello_world.py           ← 14-step full-stack smoke test
      HELLO_WORLD_REPORT.md    ← verified performance report
      live_demo/               ← real-time proof of concept (localhost, no API key)
      recursive_shutdown.py    ← three-level recursive enforcement → guaranteed shutdown
      RECURSIVE_SHUTDOWN.md    ← architecture and walkthrough
      paperclip_maximizer.py   ← canonical misaligned-AI example → 12 fields, 4 forbidden, breaker cascade
      PAPERCLIP_MAXIMIZER.md   ← scenario walkthrough and architecture
  numerail_ext/                ← survivability extension
    src/numerail_ext/survivability/
      breaker.py               ← BreakerStateMachine (5 modes, hysteretic transitions)
      transition_model.py      ← IncidentCommanderTransitionModel (13-field monotone caps)
      policy_builder.py        ← build_v5_policy_from_envelope()
      global_default.py        ← build_global_default() (30 fields, ~80 constraints)
      governor.py              ← StateTransitionGovernor (12-step lifecycle)
      contract.py              ← NumerailPolicyContract (SHA-256, chain-linked)
      hitl.py                  ← SupervisedGovernor + review profiles + audit helpers
      local_gateway.py         ← LocalApprovalGateway (development/testing)
      local_backend.py         ← LocalNumerailBackend
      validation.py            ← validate_receipt_against_grant()
      types.py                 ← shared data types and Protocols
      py.typed                 ← PEP 561 marker
    tests/                     ← 306 tests
      test_integration.py      ← 10 integration tests (full governor lifecycle)
      test_hitl_foundation.py  ← 45 HITL foundation tests
      test_hitl_supervised.py  ← 44 SupervisedGovernor tests
  numerail_learn/              ← reinforcement learning from enforcement
    src/numerail_learn/
      experience.py            ← EnforcementExperienceBuffer
      reward.py                ← EnforcementRewardShaper + presets
      adapter.py               ← SFT, DPO, PPO training adapters
      orchestrator.py          ← EnforcementRLOrchestrator
    tests/                     ← 56 tests
.github/workflows/
  ci.yml                       ← 3 jobs: core, ext, integration
  release.yml                  ← 4 jobs: verify → publish (PyPI trusted publishing)
CLAUDE.md                      ← Claude Code project context
CHANGELOG.md
README.md                      ← this file
```

---

## Design

Numerail implements a **propose-check-enforce** architecture. The AI proposes. The geometry checks. The engine enforces. Nothing reaches the world unchecked.

This is the same design primitive that governs fly-by-wire flight. When a pilot — or autopilot — commands a maneuver, the flight control computer does not pass the command to the control surfaces. It checks the command against the aerodynamic envelope: angle of attack limits, load factor boundaries, stall margins. If the command is inside the envelope, it executes. If it is outside, the computer corrects it to the nearest safe equivalent or rejects it. The pilot retains full authority within the envelope but cannot exceed it. This is why fly-by-wire aircraft do not depart controlled flight — not because pilots are perfect, but because the envelope is enforced geometrically, after the proposal and before the actuator. The same principle governs robotic workspace safety, where a safety controller checks proposed trajectories against convex collision boundaries before the motors execute.

Numerail applies this architecture to AI actuation. The AI is the pilot — it proposes a workload vector. The feasible region is the envelope — convex constraints encoding what is operationally permissible. The enforcement engine is the flight control computer — it approves, projects to the nearest feasible point, or rejects. Everything else in the system exists to make this loop auditable, adaptive, and survivable.

Four principles govern the design:

**The guarantee is the product.** Theorem 1 proves that if the engine returns APPROVE or PROJECT, the output satisfies every active constraint. The service layer, the budgets, the audit chain, the breaker suite — all exist to make this guarantee useful in production. No feature is added that weakens it.

**The AI is the subject of governance, never the governor.** The AI proposes workload values. Everything else — live telemetry, controller reserves, breaker mode, budget remaining, freshness windows — is determined by the orchestrator from server-authoritative sources the AI cannot read or write.

**The kernel is unchanged by the layers above it.** The survivability extension controls which policy the kernel enforces against. It cannot modify how enforcement works. The guarantee proved for the kernel propagates unchanged to every layer built on top of it. The HITL layer wraps the governor without modifying it. The governor wraps the kernel without modifying it. The proof chain is unbroken from the mathematical theorem to the production deployment.

**The default is denial.** Every permission must be explicitly granted through constraint geometry. An empty policy rejects everything. The system is safe by construction, not by assumption.

---

## A Note on Non-Convexity

Numerail enforces convex constraints. This covers the vast majority of real-world governance requirements: caps, budgets, joint limits, rate limits, ratio constraints, energy bounds, and coupled stability constraints are all naturally convex. Most regulation is expressed as "stay below this ceiling" or "stay within this combined envelope" — both convex by construction.

Non-convex requirements do exist — "do A or B but not both," "operate below 50 or above 100 but not between," exclusion zones, and discrete authority levels. These are the exception, not the norm. When they arise, they can be decomposed into a union of convex regions and enforced by a composition layer above the kernel: enforce the proposed vector against each convex component independently, then select the result with the smallest distance. The enforcement guarantee holds for each component because each call to `enforce()` operates on a convex region. The kernel is unchanged.

This decomposition is not built into the current repository. It is a future capability that the architecture supports without modification to the kernel, the proofs, or the extension. The foundation is convex because convexity is what makes the guarantee provable — projection is unique, the solver chain converges, and the post-check is complete. Non-convexity is handled by composition, not by weakening the foundation.

---

## Challenges

Numerail guarantees that every approved or projected output satisfies every active constraint. It does not guarantee that the constraints are correct. The gap between these two statements is where every real deployment challenge lives.

**The specification problem is the hard problem.** Writing the right constraints — translating governance intent into convex geometry that captures all relevant risk, at the right levels, with the right interactions, for the right fields — is the hardest part of deploying Numerail. A credit cap set at $5,000 when the intent was $50 will be faithfully enforced at $5,000. A policy that omits a safety-relevant dimension leaves that dimension unconstrained. A joint constraint that weights two fields equally when the domain requires asymmetric weighting will enforce the wrong tradeoff with perfect fidelity. The engine cannot detect these errors because they are not errors in the engine — they are errors in the specification.

**Projection changes the AI's proposal, and the change may not match intent.** When the engine projects an infeasible vector to the nearest feasible point, the result satisfies every constraint — that is proved. But "nearest feasible point" is a geometric operation, not a semantic one. If an AI proposes 200 GPU-seconds and the cap is 120, projection clips it to 120. That is probably what the deployer intended. But if an AI proposes a workload that violates a complex joint constraint involving GPU utilization, API calls, and parallelism simultaneously, the projected point will be the closest point in the feasible polytope — which may redistribute the workload across dimensions in ways the deployer did not anticipate. The projected action is safe by the constraint definition. Whether it is operationally sensible depends on whether the constraints correctly encode the deployer's notion of sensible. For dimensions where silent correction is dangerous, the `PROJECTION_FORBIDDEN` dimension policy exists — it forces rejection instead of projection, requiring the AI to propose a feasible action on its own.

**Breaker mode transitions depend on threshold calibration.** The breaker state machine transitions between CLOSED, THROTTLED, HALF_OPEN, SAFE_STOP, and OPEN based on a weighted overload score computed from live telemetry. The weights (0.30 GPU, 0.25 API, 0.20 DB, 0.10 queue, 0.15 error rate) and the threshold values (trip_score, reset_score, safe_stop_score) determine when the system degrades. If the trip threshold is too low, the system throttles under normal load and rejects actions that should be permitted. If it is too high, the system stays in CLOSED mode under genuine stress and permits actions that should be restricted. If the reset threshold is too close to the trip threshold, the breaker oscillates between modes. If the safe_stop threshold is too low, SAFE_STOP latches prematurely and requires manual intervention when automated recovery would have been appropriate. None of these are bugs in the breaker logic — the transitions are correct for whatever thresholds you configure. They are calibration problems, and they require operational experience with the specific infrastructure being governed.

**Budget sizing determines how long authority lasts.** The three shift budgets (GPU-seconds, external API calls, cloud mutations) deplete monotonically as the AI consumes resources. If the GPU budget is set to 3,600 seconds for an 8-hour shift, the AI can lease 450 seconds per hour on average. If the AI's workload is bursty — consuming 2,000 seconds in the first hour — it will hit budget exhaustion well before the shift ends, and every subsequent action that requires GPU time will be rejected. The budget is not wrong; the budget is exactly what was specified. But the deployer may have intended a burst-tolerant budget with per-hour sub-limits rather than a single-shift pool. Numerail provides the budget tracking and enforcement machinery. The deployer must decide the budget structure.

**Reserve sizing determines whether the governance system survives.** The control-plane reserve pattern subtracts protected capacity for the governance controller before evaluating the AI's request. If the GPU reserve is set to 30 seconds but the controller actually requires 50 seconds to complete its monitoring and enforcement cycle under load, the reserve is insufficient and the controller can be starved even though the constraints say it should be protected. The constraint will be satisfied — the 30-second reserve will be preserved. But 30 seconds is not enough. This is a specification error in the reserve value, not a failure of the enforcement engine.

**Trusted context accuracy is assumed, not verified.** The server-authoritative fields — current GPU utilization, API utilization, error rates, controller reserves, disturbance margins — are injected by the orchestrator from infrastructure telemetry. If the telemetry source reports GPU utilization as 0.30 when the actual utilization is 0.85, the enforcement engine will evaluate the AI's request against a constraint that thinks there is 55% more headroom than actually exists. The engine guarantees that the output satisfies every constraint as computed from the injected values. It does not guarantee that the injected values are accurate. Stale telemetry is partially mitigated by the three-layer freshness architecture (engine constraints, governor clock check, reservation expiry), but telemetry that is fresh and wrong — a sensor reporting an incorrect value — passes through the freshness checks and corrupts the constraint evaluation silently.

None of this is unique to Numerail. Every rule-based governance system in every regulated industry faces the same problem: building codes, aviation maintenance procedures, flight envelopes, financial compliance rules. The constraints are written by humans, and humans can write the wrong constraints. Numerail's contribution is to make the enforcement provably correct so that specification becomes the only remaining problem — and specification is a legible, auditable, iterative engineering problem with established tools, observable metrics, and a versioned policy history that supports continuous refinement.

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
| Performance benchmarks | [`packages/numerail/tests/BENCHMARK_REPORT.md`](packages/numerail/tests/BENCHMARK_REPORT.md) |
| Hello world report | [`packages/numerail/examples/HELLO_WORLD_REPORT.md`](packages/numerail/examples/HELLO_WORLD_REPORT.md) |
| Regulatory bodies | [`packages/numerail/docs/REGULATORY.md`](packages/numerail/docs/REGULATORY.md) |
| Recursive shutdown | [`packages/numerail/examples/RECURSIVE_SHUTDOWN.md`](packages/numerail/examples/RECURSIVE_SHUTDOWN.md) |
| Paperclip maximizer | [`packages/numerail/examples/PAPERCLIP_MAXIMIZER.md`](packages/numerail/examples/PAPERCLIP_MAXIMIZER.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |

---

## License

MIT — see [`packages/numerail/LICENSE`](packages/numerail/LICENSE).
