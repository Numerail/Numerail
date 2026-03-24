# Numerail v5.0.0 — Mathematical Proof of the Enforcement Guarantee

## Notation

Let:
- x ∈ ℝⁿ denote the proposed vector (finite, validated at entry)
- F = {C₁, C₂, ..., Cₘ} denote the active feasible region, a finite collection of convex constraints
- Cⱼ denote a single convex constraint with membership test `is_satisfied(z, τ)` and violation function `evaluate(z)`
- τ > 0 denote the configured tolerance (default 10⁻⁶)
- `check(z, F, τ)` = ∀j ∈ {1,...,m}: Cⱼ.is_satisfied(z, τ) — the combined feasibility checker
- `enforce(x, F, τ)` → (r, y) where r ∈ {APPROVE, PROJECT, REJECT} and y ∈ ℝⁿ ∪ {⊥}
- `project(x, F, τ)` → (y, passed) where y ∈ ℝⁿ and passed ∈ {true, false}

We write y ∈ F_τ to mean check(y, F, τ) = true, i.e., ∀j: Cⱼ.evaluate(y) ≤ τ.

---

## Axiom 1 (Checker Correctness)

For each constraint type, `is_satisfied(z, τ)` correctly characterizes the τ-relaxed constraint set:

**LinearConstraints:** `is_satisfied(z, τ)` returns `np.all(A @ z <= b + τ)`.

This is equivalent to `evaluate(z) ≤ τ` because `evaluate(z) = max(Az − b)`, and `max(v) ≤ τ` iff `all(vⱼ ≤ τ)`.

**QuadraticConstraint:** `is_satisfied(z, τ)` returns `evaluate(z) ≤ τ` where `evaluate(z) = z'Qz + a'z − b`.

Q is verified PSD at construction (eigenvalue check rejects non-PSD), guaranteeing the sublevel set {z : z'Qz + a'z ≤ b} is convex.

**SOCPConstraint:** `is_satisfied(z, τ)` returns `evaluate(z) ≤ τ` where `evaluate(z) = ‖Mz + q‖ − (c'z + d)`.

**PSDConstraint:** `is_satisfied(z, τ)` returns `evaluate(z) ≤ τ` where `evaluate(z) = −λ_min(A₀ + Σ zᵢAᵢ)`.

Axiom 1 states: for each Cⱼ, `Cⱼ.is_satisfied(z, τ) = true` ↔ `Cⱼ.evaluate(z) ≤ τ`.

This is verified by direct inspection of each implementation.

---

## Lemma 1 (Combined Checker Correctness)

`check(z, F, τ) = true` if and only if `∀j ∈ {1,...,m}: Cⱼ.evaluate(z) ≤ τ`.

**Proof.** `check` is implemented as:

```python
def is_feasible(self, x, tol=1e-6):
    return all(c.is_satisfied(x, tol) for c in self._constraints)
```

By Axiom 1, each `c.is_satisfied(x, tol)` is true iff `c.evaluate(x) ≤ tol`. Python's `all()` returns true iff every element is true. Therefore `is_feasible(x, tol)` returns true iff `∀j: Cⱼ.evaluate(x) ≤ tol`. ∎

---

## Lemma 2 (Project Post-Check)

If `project(x, F, τ)` returns (y, true), then `check(y, F, τ) = true`.

**Proof.** The `project()` function has exactly five return points:

| Return | Condition | postcheck_passed |
|---|---|---|
| R1 | `region.is_feasible(x, tol)` | True |
| R2 | `region.is_feasible(boxed, tol)` | True |
| R3 | `region.is_feasible(y_slsqp, tol)` | True |
| R4 | `region.is_feasible(y_dyk, tol)` | True |
| R5 | all solvers failed | False |

For R1–R4, `postcheck_passed = True` and the returned vector is the argument to `is_feasible` that returned True. By Lemma 1, `check(y, F, τ) = true`.

For R5, `postcheck_passed = False`. This path is excluded by the hypothesis `passed = true`.

In every case where `passed = true`, the returned vector y satisfies `check(y, F, τ)`. ∎

---

## Lemma 3 (Emit Path Invariant)

The helper function `_out(result, enforced, ...)` inside `enforce()` satisfies:

> If `result ∈ {APPROVE, PROJECT}`, then `check(enforced, F_effective, τ) = true`.

**Proof.** The implementation of `_out` contains:

```python
if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
    if not effective.is_feasible(enforced, cfg.solver_tol):
        raise AssertionError(...)
```

This check fires before the `EnforcementOutput` is constructed. It uses an explicit `raise` (not a Python `assert`) so it cannot be stripped by `python -O`. If `is_feasible` returns False, the `AssertionError` is raised and no output is returned. Therefore, any successfully returned output with `result ∈ {APPROVE, PROJECT}` has `is_feasible(enforced, solver_tol) = true`, which by Lemma 1 means `check(enforced, F_effective, τ) = true`. ∎

---

## Theorem 1 (Enforcement Soundness)

For any proposed vector x ∈ ℝⁿ, any active feasible region F, and any enforcement configuration:

> If `enforce(x, F)` returns (r, y) with r ∈ {APPROVE, PROJECT}, then y ∈ F_τ.

That is: ∀j ∈ {1,...,m}: Cⱼ.evaluate(y) ≤ τ.

**Proof.** The `enforce()` function has exactly 8 return points:

| Return | Result | Enforced vector | Guard condition |
|---|---|---|---|
| R1 | APPROVE | x | `effective.is_feasible(x)` is True |
| R2 | REJECT | x | hard wall violated |
| R3 | REJECT | x | mode = "reject" |
| R4 | REJECT | x | `not proj.postcheck_passed` |
| R5 | REJECT | x | forbidden dimension changed |
| R6 | REJECT | x | routing hard reject |
| R7 | REJECT | x | hybrid distance exceeded |
| R8 | PROJECT | proj.point | all operational filters passed |

Returns R2–R7 have `result = REJECT`. These are excluded by the hypothesis `r ∈ {APPROVE, PROJECT}`.

**Case R1 (APPROVE):** The guard is `effective.is_feasible(x)`, which returned True. The enforced vector is x. By Lemma 1, `check(x, F, τ) = true`. Additionally, `_out` is called with `result = APPROVE` and `enforced = x`, so by Lemma 3, the assertion confirms `is_feasible(x, τ)`. Therefore y = x ∈ F_τ.

**Case R8 (PROJECT):** This return is reached only when:
1. `effective.is_feasible(x)` was False (otherwise R1 would have fired)
2. No hard wall was violated (otherwise R2)
3. Mode ≠ "reject" (otherwise R3)
4. `proj.postcheck_passed` is True (otherwise R4)
5. All operational filters passed (otherwise R5, R6, or R7)

From condition 4: `proj.postcheck_passed = true`. By Lemma 2, `check(proj.point, F, τ) = true`. The enforced vector is `proj.point`. Additionally, `_out` is called with `result = PROJECT` and `enforced = proj.point`, so by Lemma 3, the assertion confirms `is_feasible(proj.point, τ)`. Therefore y = proj.point ∈ F_τ.

In both cases, y ∈ F_τ. ∎

---

## Theorem 2 (Fail-Closed Rejection)

If all solvers in the chain produce candidates that fail the post-check, the result is REJECT.

**Proof.** When all solvers fail, `project()` returns `ProjectionResult(x.copy(), 0, "none", False, False)`. In `enforce()`, the check `if not proj.postcheck_passed` (return R4) fires, producing REJECT. No other code path between R4 and R8 can produce APPROVE or PROJECT — returns R5, R6, R7 are all REJECT, and R8 is unreachable because it requires `proj.postcheck_passed = True`. ∎

---

## Theorem 3 (Hard-Wall Dominance)

If any hard-wall constraint is violated by x, the result is REJECT and no solver is invoked.

**Proof.** The hard-wall check (return R2) occurs before the solver invocation (Step 4). If `cfg.hard_wall_constraints & frozenset(violated_names)` is non-empty, return R2 fires with `result = REJECT` and `solver = "none"`. The solver chain is never called. ∎

---

## Theorem 4 (Forbidden-Dimension Safety)

If `enforce()` would need to change a forbidden dimension to achieve feasibility, the result is REJECT.

**Proof.** Return R5 checks each dimension policy. For any dimension `i` with policy `PROJECTION_FORBIDDEN`, if `|proj.point[i] − x[i]| > 10⁻⁹`, return R5 fires with `result = REJECT`. The projected point (which may be feasible) is not emitted. ∎

---

## Theorem 5 (Budget Monotonicity)

Under nonnegative consumption mode, the budget-tightened feasible region is monotone non-expanding: F_{t+1} ⊆ F_t.

**Proof.** Under nonnegative mode, consumption = max(0, enforced_value) × weight ≥ 0 for every enforcement. The cumulative consumed total is non-decreasing. The remaining budget R_t = initial − consumed_t is non-increasing. The targeted constraint row's bound is set to R_t, so b_{t+1} ≤ b_t. The half-space {z : a'z ≤ b_{t+1}} ⊆ {z : a'z ≤ b_t}. All other constraints are unchanged. Therefore F_{t+1} ⊆ F_t. ∎

---

## Theorem 6 (Rollback Restoration)

After `rollback(action_id)`, the consumed total for each affected budget is reduced by exactly the stored delta, and the remaining budget is recomputed as R = initial − consumed.

**Proof.** The `rollback()` method retrieves the stored delta dict for the action_id, then subtracts each delta from the corresponding budget's consumed total: `consumed[name] -= delta`. The remaining is `initial - consumed[name]`. This correctly removes the action's contribution regardless of subsequent actions, because each action's delta is recorded independently at enforcement time and the subtraction is applied to the current cumulative total. Rollback is idempotent: the action's delta is removed from the `_action_deltas` dict on first rollback, so a second call returns False. ∎

---

## Theorem 7 (Monotone Self-Limits)

Under the following assumptions:
- (A1) Only an authorized policy authority can create or activate policy versions.
- (A2) The agent's API key carries only the enforce scope.
- (A3) The enforcement mode is "reject".

If the policy upper bounds are updated to the approved capability vector after each successful enforcement, then any subsequent proposal that increases any bound is rejected.

**Proof.** After an approved self-throttle to vector c', the orchestrator creates a new policy version with upper bounds c'ᵢ for each dimension i. In reject mode, any infeasible proposal is rejected without projection (return R3 in `enforce()`). A proposal c'' with c''ᵢ > c'ᵢ for any i violates the upper bound on dimension i. By A2, the agent cannot create or activate versions to loosen these bounds. ∎

---

## Theorem 8 (Audit-Chain Integrity)

Under SHA-256 collision resistance, any modification, insertion, or deletion of an audit record is detectable by chain verification.

**Proof.** Each record includes `prev_hash` (the hash of the preceding record) within the hashed payload: `h = SHA256(canonical_json(record))` where `record["prev_hash"] = h_{i-1}`. Modifying record i changes h_i, which breaks the `prev_hash` link in record i+1. Inserting a record between i and i+1 changes the `prev_hash` of the inserted record's successor. Deleting record i breaks the chain at record i+1 (its `prev_hash` no longer matches). The `verify()` method walks the chain from the first retained record, recomputes each hash, and compares. Under collision resistance (finding x ≠ y with SHA256(x) = SHA256(y) is computationally infeasible), any tampering is detected. ∎

---

## Theorem 9 (Passthrough and Idempotence)

**(a) Passthrough.** If x ∈ F_τ, then `enforce(x, F)` returns (APPROVE, x) with distance 0.

**Proof.** Return R1 checks `effective.is_feasible(x)`. Since x ∈ F_τ, this returns True. The function returns APPROVE with `enforced = x` and `distance = 0`. ∎

**(b) Idempotence.** If `enforce(x, F)` returns (r, y) with r ∈ {APPROVE, PROJECT}, then `enforce(y, F)` returns (APPROVE, y) with distance 0.

**Proof.** By Theorem 1, y ∈ F_τ. By part (a), `enforce(y, F)` returns (APPROVE, y) with distance 0. ∎

---

## Corollary (Solver Independence)

The guarantee of Theorem 1 holds for any solver implementation, including adversarial ones.

**Proof.** Theorem 1's proof never assumes anything about the solver's behavior. The solver appears only in Step 4 (`project(x, F, τ)`). By Lemma 2, any vector returned with `postcheck_passed = True` satisfies `check(y, F, τ)`. If the solver produces garbage, `postcheck_passed = False`, and return R4 (REJECT) fires. The guarantee follows from the post-check and the control flow, not from any solver property. ∎

---

## Corollary (Proposer Independence)

The guarantee of Theorem 1 holds for any proposed vector x, including adversarial ones.

**Proof.** Theorem 1 is universally quantified over x. The proposed vector enters the function, is validated for finiteness and dimension, and is then either approved (if feasible), projected (if a feasible correction exists and passes all filters), or rejected. No property of x beyond finiteness and correct dimension is assumed. NaN and Inf inputs are rejected by `_validate_vector()` before enforcement begins. ∎

---

## Summary of Proof Structure

```
Axiom 1 (Checker Correctness)
    │
    ▼
Lemma 1 (Combined Checker = conjunction of individual checkers)
    │
    ├──► Lemma 2 (project postcheck_passed=True ⟹ check(y)=True)
    │
    ├──► Lemma 3 (emit path assert ⟹ check(enforced)=True)
    │
    ▼
Theorem 1 (APPROVE/PROJECT ⟹ y ∈ F_τ)
    │
    ├──► Theorem 2 (all solvers fail ⟹ REJECT)
    ├──► Corollary (solver independence)
    ├──► Corollary (proposer independence)
    └──► Theorem 9b (idempotence, via Theorem 1 + Theorem 9a)

Theorem 3 (hard-wall dominance) — independent of Theorem 1
Theorem 4 (forbidden-dimension safety) — independent of Theorem 1
Theorem 5 (budget monotonicity) — independent of Theorem 1
Theorem 6 (rollback restoration) — independent of Theorem 1
Theorem 7 (monotone self-limits) — depends on external assumptions A1-A3
Theorem 8 (audit integrity) — depends on SHA-256 collision resistance
Theorem 9a (passthrough) — independent, used by 9b
```

The central result is Theorem 1. It depends on Axiom 1 (checker correctness), Lemma 1 (conjunction), Lemma 2 (project post-check), and Lemma 3 (emit assertion). Everything else is either a consequence of Theorem 1 or an independent operational property.

---

## Correspondence to Code

| Proof element | Code location |
|---|---|
| Axiom 1 | `ConvexConstraint.is_satisfied`, `LinearConstraints.is_satisfied`, `QuadraticConstraint.is_satisfied`, `SOCPConstraint.is_satisfied`, `PSDConstraint.is_satisfied` |
| Lemma 1 | `FeasibleRegion.is_feasible` |
| Lemma 2 | `project()` — 5 return points, each guarded by `region.is_feasible()` |
| Lemma 3 | `_out()` inside `enforce()` — assert statement |
| Theorem 1, R1 | `enforce()` line: `if effective.is_feasible(x): return _out(APPROVE, x, ...)` |
| Theorem 1, R8 | `enforce()` line: `return _out(PROJECT, proj.point, ...)` — reached only after `proj.postcheck_passed` |
| Theorem 2 | `enforce()` line: `if not proj.postcheck_passed: return _out(REJECT, ...)` |
| Theorem 3 | `enforce()` line: hard wall check before solver invocation |
| Theorem 4 | `enforce()` line: dimension policy loop after post-check |
| Theorem 5 | `BudgetTracker.record_consumption` with nonnegative mode |
| Theorem 6 | `BudgetTracker.rollback` |
| Theorem 7 | `NumerailSystem` + reject mode + external version control |
| Theorem 8 | `AuditChain.append` and `AuditChain.verify` |
| Theorem 9 | Theorem 1 + `is_feasible` check at R1 |
