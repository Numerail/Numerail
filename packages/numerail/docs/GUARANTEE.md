# The Guarantee

Numerail v5.0.0 — Deterministic Geometric Enforcement for AI Actuation Safety

---

## What Numerail guarantees

Numerail makes one mathematical promise. If the enforcement engine returns APPROVE or PROJECT, the enforced output vector satisfies every active constraint in the current feasible region. This holds for all proposed inputs, all constraint type combinations, and all solver implementations. The solver is untrusted. The guarantee comes from the post-check, not the solver.

Formally:

> r ∈ {APPROVE, PROJECT} ⟹ ∀ c ∈ F_t.constraints : c.evaluate(y) ≤ τ

Where x is the proposed vector, F_t is the active feasible region at enforcement time t, y is the enforced output, τ is the configured tolerance (default 10⁻⁶), and r is the enforcement decision.

If the engine cannot verify admissibility, it rejects. Nothing reaches the world as "approved" or "projected" without passing the combined feasibility checker. This is fail-closed by construction: inability to verify is treated as inadmissibility.

## Why the guarantee holds

The guarantee is structural, not statistical. It does not depend on testing, training data, or the model behaving well. It depends on three things.

First, the control flow of `enforce()`. There is exactly one APPROVE return and one PROJECT return in the function. The APPROVE return executes only inside `if effective.is_feasible(x)`. The PROJECT return executes only after `proj.postcheck_passed` is confirmed True, and `postcheck_passed` is set True only when `region.is_feasible(y, tol)` returns True inside the `project()` function. There are six REJECT returns, any of which can fire at any point. Steps 6 through 8 of the pipeline can only convert a PROJECT into a REJECT — they cannot introduce a new APPROVE or PROJECT, and they cannot mutate the checked vector.

Second, the `is_feasible()` method. This calls `is_satisfied(x, tol)` on every constraint in the region via Python's `all()`. A single failing constraint is sufficient for rejection. The conjunction is complete.

Third, a defense-in-depth check in the emit path:

```python
if result in (APPROVE, PROJECT):
    if not effective.is_feasible(enforced, cfg.solver_tol):
        raise AssertionError(
            "Numerail invariant violated: emitted vector must satisfy "
            "the exact combined checker for the active feasible region. "
            "This check should never fire in correctly functioning code."
        )
```

This check is redundant with the control flow — it should never fire. It uses an explicit `raise` rather than a bare `assert` so it cannot be stripped by `python -O`. Its purpose is to make the invariant visible to code reviewers and to catch implementation bugs if someone modifies the enforcement logic incorrectly.

The guarantee is proved in `proof/PROOF.md` (Axiom 1, Lemmas 1–3, Theorems 1–9, 2 Corollaries). The proof is independently verified by `proof/verify_proof.py` (3,732 structural and property checks) and `tests/test_guarantee.py` (46 certification tests across 7 categories).

## The two forms of the guarantee

The formal guarantee states exact membership:

> r ∈ {APPROVE, PROJECT} ⟹ y ∈ F_t

The numerical implementation guarantee, as realized in Python with IEEE 754 double-precision arithmetic, states membership up to tolerance:

> r ∈ {APPROVE, PROJECT} ⟹ ∀ c ∈ F_t.constraints : c.evaluate(y) ≤ τ

The tolerance τ bridges the abstract mathematical model and the floating-point implementation. At the default τ = 10⁻⁶, this provides approximately 10 orders of magnitude above machine epsilon (~10⁻¹⁶). In certification testing, the worst-case observed violation was 8.51 × 10⁻¹¹ — roughly ten thousand times below the tolerance bound.

Both the APPROVE and PROJECT paths use `is_feasible(x, cfg.solver_tol)`, where `solver_tol` is configurable (default 10⁻⁶). The same tolerance value is used by the R1 APPROVE gate, the `project()` post-check, and the defense-in-depth assertion in `_out()` — all three checks are consistent under any configured tolerance. If `solver_tol` is increased above 10⁻⁶, all three paths accept vectors with proportionally larger violations. This is a deliberate design choice: it trades precision for solver convergence reliability in numerically difficult constraint compositions.

## What the guarantee does not establish

The guarantee does not establish correctness of the constraint specification. It proves y ∈ F_t. It does not prove that F_t is the right region. A credit cap of $5,000 when the governance intent was $50 will be faithfully enforced at $5,000. The engine enforces whatever geometry you define, whether or not that geometry captures your intent.

The guarantee does not establish truth of external context. If budget bounds or trusted field values are derived from external data (fraud scores, market prices, system quotas), the guarantee holds for F_t as computed from those values, not for what F_t "should" be if the external data were different.

The guarantee does not establish optimality of the projection. The enforced vector y is feasible, but it may not be the nearest feasible point to the proposal x. The solver chain attempts to minimize ‖y − x‖, but only feasibility is guaranteed, not minimum-distance optimality.

The guarantee does not establish anything about the model's internal behavior. Numerail constrains the output, not the intent. A model that repeatedly proposes inadmissible vectors wastes capacity, and the metrics detect this, but the enforcement layer does not modify the model.

## What the guarantee depends on

Three conditions, all explicit:

**C1. Checker correctness.** Each `is_satisfied(x, τ)` implementation correctly characterizes {z : evaluate(z) ≤ τ}. For LinearConstraints, `is_satisfied` returns `np.all(Ax ≤ b + τ)`. For all other types, it returns `evaluate(x) ≤ τ`. Both are direct numerical evaluations of the constraint definition. This is verified structurally by `verify_proof.py` and tested empirically for all four concrete types.

**C2. Finite arithmetic.** IEEE 754 double-precision may introduce rounding errors. The tolerance τ absorbs these. At τ = 10⁻⁶, a point with true violation ε and rounding error δ passes the check when ε + δ ≤ τ.

**C3. No concurrent mutation.** The `enforce()` function is pure — no shared mutable state. `NumerailSystem.enforce()` holds `_enforce_lock` for the entire operation, preventing concurrent budget or region mutation between the feasibility check and the return.

## What the guarantee does not depend on

The solver. The guarantee is solver-independent. The solver (box clamp, SLSQP, Dykstra) proposes a candidate. The post-check verifies it. If the solver is wrong, the result is REJECT. Even an adversarial solver cannot violate the guarantee.

The proposed vector. The guarantee holds for all inputs, including adversarial ones. NaN and Inf inputs are rejected before enforcement begins.

The constraint type mix. Linear, quadratic, SOCP, and PSD constraints can be composed in any combination. The guarantee holds for all compositions because `is_feasible` checks every constraint.

The model. The guarantee is proposer-independent. It quantifies over proposed vectors, not over model internals.

---


---

## Verification

The guarantee is proved in `proof/PROOF.md` and independently verified by two test suites:

- `proof/verify_proof.py` — 3,732 structural and property checks against the engine code. Validates every axiom, lemma, theorem, and corollary.
- `tests/test_guarantee.py` — 46 certification tests across 7 categories: structural verification, formal property tests (mirroring Theorems 1–9), constraint type coverage, adversarial probes, randomized stress tests, enforcement mode coverage, and tolerance precision.

To verify independently:

```
pip install -e .
pytest tests/test_guarantee.py -v
python proof/verify_proof.py
```

The worst-case observed violation across all tests is below 10⁻¹⁰ against a tolerance bound of 10⁻⁶.

A deployment may additionally encode control-survivability constraints inside the feasible region — for example, reserving GPU, API, and parallelism capacity for the governance controller itself. In that case, Numerail's existing guarantee implies that any approved or projected action also preserves the encoded control reserve. This stronger property comes from the policy design, not from a different engine theorem. See `docs/DEPLOYMENT.md` and `examples/ai_circuit_breaker.py` for the pattern.
