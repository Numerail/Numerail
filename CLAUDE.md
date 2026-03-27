# Numerail — Project Brief for Claude Code

## What Numerail Is

Numerail is a **deterministic geometric enforcement kernel for AI actuation safety**. When an AI proposes a numerical action (token budget, GPU lease, API-call grant, trade size, voltage setpoint), Numerail checks it against convex geometric constraints and returns one of three outcomes:

- **APPROVE** — the proposal already satisfies every constraint
- **PROJECT** — the proposal was corrected to the nearest feasible point
- **REJECT** — the proposal was blocked

The solver is untrusted. The guarantee comes from a post-check that runs after every solver call.

## The Enforcement Guarantee

**This is the central invariant of the entire project. It must never be weakened.**

```
r ∈ {APPROVE, PROJECT}  ⟹  ∀ c ∈ F_t : c.evaluate(y) ≤ τ
```

If `enforce()` returns APPROVE or PROJECT, the enforced output vector satisfies every active constraint to within tolerance τ (default 10⁻⁶). This holds for:
- All proposed inputs, including adversarial ones
- All constraint type combinations (linear, quadratic, SOCP, PSD)
- All solver implementations, including adversarial solvers

The system is **fail-closed**: if no solver produces a verified feasible point, the result is REJECT. Inability to verify is treated as inadmissibility.

The guarantee is proved in `packages/numerail/proof/PROOF.md` (Axiom 1, Lemmas 1–3, Theorems 1–9, 2 Corollaries) and independently verified by:
- `packages/numerail/proof/verify_proof.py` — 3,732 structural/property checks
- `packages/numerail/tests/test_guarantee.py` — 46 certification tests across 7 categories
- `packages/numerail/tests/test_mathematical_guarantees.py` — 99 guarantee analysis tests (one per proof claim)

## Repository Layout

```
numerail-repo/                       ← repository root (tagged v5.0.0 / ext v0.4.0)
  packages/
    numerail/                        ← core enforcement kernel (v5.0.0)
      src/numerail/
        engine.py                    ← THE mathematical kernel (single file, no external deps beyond numpy/scipy)
        parser.py                    ← policy parser + lint_config
        service.py                   ← production runtime service
        local.py                     ← in-memory local mode + DefaultTimeProvider + SystemEnforcementResult
        protocols.py                 ← typed Protocol interfaces + TrustedContextProvider
        errors.py                    ← production-layer exceptions
      tests/                         ← 265 tests total
        test_guarantee.py            ← 46 guarantee certification tests (most critical)
        test_mathematical_guarantees.py ← 99 guarantee analysis tests (one per proof claim)
        test_trusted_context.py      ← 10 trusted context injection tests
      proof/
        PROOF.md                     ← mathematical proof (Axiom 1, Lemmas 1-3, Theorems 1-9)
        verify_proof.py              ← machine-verifiable proof checker (3,732 checks)
        Guarantee.v                  ← Rocq/Coq machine-checked formalization (11 proofs, 0 Admitted)
        Guarantee.lean               ← Lean 4 machine-checked formalization (12 proofs, 0 sorry)
      docs/
        DEVELOPER_GUIDE.md           ← start here for any development work
        GUARANTEE.md                 ← full guarantee specification
        SPECIFICATION.md             ← how to write constraint specifications
        DEPLOYMENT.md                ← production deployment guide
        REFERENCE.md                 ← API reference
      examples/
        hello_world.py               ← 14-step full-stack smoke test (all theorems, both packages)
        HELLO_WORLD_REPORT.md        ← verified performance report for hello_world.py
        ai_resource_governor.py      ← base AI governance example (8-field policy, all 4 constraint types)
        ai_circuit_breaker.py        ← control-plane reserve pattern
        autonomous_agent_governor.py ← 20-step simulation: breaker transitions, budget depletion, rollback
        rest_api_server.py           ← FastAPI server wrapping NumerailSystemLocal (3 endpoints)
        rest_api_client.py           ← stdlib-only client exercising all three server endpoints
    numerail_ext/                    ← survivability extension (v0.4.0), requires numerail ≥ 5.0.0
      src/numerail_ext/survivability/
        breaker.py                   ← BreakerStateMachine
        transition_model.py          ← IncidentCommanderTransitionModel
        policy_builder.py            ← build_v5_policy_from_envelope()
        global_default.py            ← build_global_default() policy pack
        governor.py                  ← StateTransitionGovernor
        contract.py                  ← NumerailPolicyContract (content-addressable, chain-linked)
        local_backend.py             ← LocalNumerailBackend
        validation.py                ← validate_receipt_against_grant()
        types.py                     ← shared data types and Protocols
        hitl.py                      ← HumanReviewTriggers, SupervisedGovernor, PendingAction, SupervisedStepResult
        local_gateway.py             ← LocalApprovalGateway (in-process gateway for testing)
      tests/                         ← 306 tests
        test_integration.py          ← 10 integration tests (full governor lifecycle, cross-stack)
        test_hitl_foundation.py      ← 45 HITL foundation tests (types, triggers, gateway)
        test_hitl_supervised.py      ← 44 SupervisedGovernor tests (TOCTOU, decisions, audit, expiry)
  .github/workflows/
    ci.yml                           ← CI: checkout@v5, setup-python@v6, Python 3.9–3.12
    release.yml                      ← PyPI trusted publishing on v* tag push
  CHANGELOG.md
  README.md
```

## How to Run Tests

```bash
# Core — full test suite (253 tests)
cd packages/numerail && pytest tests/ -v

# Core — guarantee certification only (46 tests, most critical)
cd packages/numerail && pytest tests/test_guarantee.py -v

# Core — mathematical guarantee analysis (99 tests, one per proof claim)
cd packages/numerail && pytest tests/test_mathematical_guarantees.py -v

# Core — machine-verifiable proof checker (3,732 checks)
cd packages/numerail && python proof/verify_proof.py

# Extension — full test suite (306 tests)
cd packages/numerail_ext && pytest tests/ -v

# Extension — integration tests only (10 tests, full governor lifecycle)
cd packages/numerail_ext && pytest tests/test_integration.py -v

# Extension — HITL foundation tests (45 tests)
cd packages/numerail_ext && pytest tests/test_hitl_foundation.py -v

# Extension — SupervisedGovernor tests (44 tests)
cd packages/numerail_ext && pytest tests/test_hitl_supervised.py -v

# Full stack — single-command smoke test (14 steps, all theorems exercised)
python packages/numerail/examples/hello_world.py
```

Install before testing:
```bash
cd packages/numerail && pip install -e .
cd packages/numerail_ext && pip install -e .
```

## Common Claude Code Tasks — Safe vs. Proof-Checker-Required

### Safe to edit without running the proof checker

These changes live outside the enforcement control flow. Run the full test suite
(`pytest tests/ -v`) afterward, but `verify_proof.py` and `test_guarantee.py`
are not strictly required.

| What | Why it's safe |
|------|---------------|
| `examples/` files | Demonstration code; no path into the kernel's enforcement logic |
| `docs/` files | Documentation only |
| `service.py`, `local.py`, `errors.py` | Production / convenience layers that call `engine.py` but don't modify it |
| `parser.py` | Config parsing; the kernel validates the parsed config itself |
| `numerail_ext/` (any file) | Extension layer; controls *which* policy the kernel enforces, not *how* |
| `.github/workflows/` | CI / CD plumbing |
| `pyproject.toml`, `README.md`, `CHANGELOG.md` | Packaging and docs |
| New tests that only call the public API | Tests cannot weaken the guarantee |

### Always run `proof/verify_proof.py` and `tests/test_guarantee.py` after these

Any change to `engine.py` — no exceptions. Specifically:

| Touch point in `engine.py` | Risk |
|-----------------------------|------|
| `enforce()` control flow (APPROVE / PROJECT / REJECT branches) | Could create a path that returns APPROVE/PROJECT without a feasibility check — directly breaks Theorem 1 |
| `_out()` (the emit-path function) | The defense-in-depth `raise AssertionError` must fire before output construction; reordering breaks Lemma 3 |
| `project()` and `postcheck_passed` | Lemma 2 depends on `postcheck_passed = True` being set only after `is_feasible` confirms the projected point |
| `FeasibleRegion.is_feasible` or any `is_satisfied` method | Lemma 1 requires the conjunction of *all* constraints; skipping or caching any one of them breaks the guarantee |
| Solver tolerance (`solver_tol`) or the `τ`-relaxation value | Widening τ relaxes the guarantee; tightening it can cause false REJECTs |
| Any new constraint type added to the kernel | Must implement `is_satisfied` correctly for Axiom 1 to hold for that type |

**Command to run after any `engine.py` edit:**
```bash
cd packages/numerail
python proof/verify_proof.py && pytest tests/test_guarantee.py -v
```

### Things to never do regardless of where the edit is

- Add a code path in `engine.py` that returns APPROVE or PROJECT without calling `effective.is_feasible()` first.
- Convert the explicit `raise AssertionError(...)` in `_out()` to a bare `assert` statement (strips under `python -O`).
- Add an import to `engine.py` that pulls in `service.py`, `local.py`, or any production-layer module.
- Let model/agent output influence constraint construction, tolerance values, or trusted-field injection.
- Add fallback logic that permits an action when constraints are missing or ambiguous.

## Architectural Invariants — Never Violate These

### 1. The post-check is the trust boundary
The `enforce()` function in `engine.py` has exactly one APPROVE return (gated by `effective.is_feasible(x)`) and exactly one PROJECT return (reachable only after `proj.postcheck_passed = True`). There are six REJECT returns. **Never add a code path that returns APPROVE or PROJECT without a preceding feasibility check.**

### 2. The defense-in-depth assertion uses `raise`, not `assert`
Inside `_out()` in `engine.py`:
```python
if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
    if not effective.is_feasible(enforced, cfg.solver_tol):
        raise AssertionError(...)
```
This uses an explicit `raise` so it **cannot be stripped by `python -O`**. Never convert this to a bare `assert` statement.

### 3. The kernel is unchanged by the layers above it
`engine.py` is a single file with no dependencies beyond numpy and scipy. `numerail_ext` wraps the kernel but cannot modify how enforcement works. The breaker suite controls *which policy* the kernel enforces against — it cannot change *how enforcement works*. Never add imports to `engine.py` that pull in production-layer code.

### 4. The AI is the subject of governance, never the governor
The AI proposes numerical values. Everything else — trusted fields, breaker mode, envelope ceilings, budget remaining, freshness window — is determined by the orchestrator from server-authoritative sources the AI cannot read or write. Never design a path where model output influences constraint construction or tolerance values.

### 5. The default is denial
Every permission must be explicitly granted through constraint geometry. Do not add fallback behavior that permits actions when constraints are ambiguous or missing.

### 6. `is_feasible` calls `is_satisfied` on every constraint
`FeasibleRegion.is_feasible` uses Python's `all()` over every constraint. A single failing constraint is sufficient for rejection. Never short-circuit or cache this check in a way that could skip a constraint.

### 7. The emit-path assert fires before output construction
The feasibility re-check in `_out()` fires before the `EnforcementOutput` object is constructed. Never reorder this so that a partially-constructed output could escape if the check raises.

## Proof Structure (for reference when editing engine.py)

```
Axiom 1 (each is_satisfied correctly characterizes the τ-relaxed set)
    ▼
Lemma 1 (is_feasible = conjunction of all is_satisfied)
    ├──► Lemma 2 (project() postcheck_passed=True ⟹ is_feasible(y)=True)
    ├──► Lemma 3 (_out() assert ⟹ is_feasible(enforced)=True)
    ▼
Theorem 1 (APPROVE/PROJECT ⟹ y ∈ F_τ)  ← THE GUARANTEE
    ├──► Corollary: solver independence
    └──► Corollary: proposer independence

Theorem 2: all solvers fail ⟹ REJECT
Theorem 3: hard-wall violated ⟹ REJECT before solver
Theorem 4: forbidden dimension changed ⟹ REJECT
Theorem 5: budget monotonicity (non-expanding feasible region)
Theorem 6: rollback restoration
Theorem 7: monotone self-limits (requires external assumptions A1–A3)
Theorem 8: audit-chain integrity (SHA-256)
Theorem 9: passthrough (x ∈ F_τ ⟹ APPROVE) + idempotence
```

Any edit to `engine.py` that touches the enforcement control flow, `is_feasible`, `is_satisfied`, `project()`, or `_out()` must be verified against this proof structure. Run `proof/verify_proof.py` and `tests/test_guarantee.py` after every such change.
