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
- `packages/numerail/tests/test_guarantee.py` — 45 certification tests across 7 categories

## Repository Layout

```
numerail-repo/                       ← repository root
  packages/
    numerail/                        ← core enforcement kernel (v5.0.0)
      src/numerail/
        engine.py                    ← THE mathematical kernel (single file, no external deps beyond numpy/scipy)
        parser.py                    ← policy parser + lint_config
        service.py                   ← production runtime service
        local.py                     ← in-memory local mode
        protocols.py                 ← typed Protocol interfaces
        errors.py                    ← production-layer exceptions
      tests/                         ← 153 tests total
        test_guarantee.py            ← 45 guarantee certification tests (most critical)
      proof/
        PROOF.md                     ← mathematical proof
        verify_proof.py              ← machine-verifiable proof checker (3,732 checks)
      docs/
        DEVELOPER_GUIDE.md           ← start here for any development work
        GUARANTEE.md                 ← full guarantee specification
        SPECIFICATION.md             ← how to write constraint specifications
        DEPLOYMENT.md                ← production deployment guide
        REFERENCE.md                 ← API reference
      examples/
        ai_resource_governor.py      ← base AI governance example
        ai_circuit_breaker.py        ← control-plane reserve pattern
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
      tests/                         ← 207 tests
  .github/workflows/                 ← CI
  CHANGELOG.md
  README.md
```

## How to Run Tests

```bash
# Core — full test suite (153 tests)
cd packages/numerail && pytest tests/ -v

# Core — guarantee certification only (45 tests, most critical)
cd packages/numerail && pytest tests/test_guarantee.py -v

# Core — machine-verifiable proof checker (3,732 checks)
cd packages/numerail && python proof/verify_proof.py

# Extension — full test suite (207 tests)
cd packages/numerail_ext && pytest tests/ -v
```

Install before testing:
```bash
cd packages/numerail && pip install -e .
cd packages/numerail_ext && pip install -e .
```

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
