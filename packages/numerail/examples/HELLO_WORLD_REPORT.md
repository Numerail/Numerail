# Hello World Performance Report

`packages/numerail/examples/hello_world.py` — full-stack end-to-end exercise of every major Numerail capability.

Run from the repository root with no arguments:

```bash
python packages/numerail/examples/hello_world.py
```

---

## Execution Summary

| Run | Result | Wall Time |
|-----|--------|-----------|
| Full script (`python packages/numerail/examples/hello_world.py`) | **14/14 steps passed** | ~33 s |
| Steps 1–13 (combined kernel + extension work) | All assertions pass | ~0.8 s |
| Step 14 (proof checker subprocess) | 3,732 / 3,732 assertions pass | ~32 s |

---

## Step-by-Step Results

| Step | Name | Result | Notes |
|------|------|--------|-------|
| 1 | The Guarantee | PASS | APPROVE on feasible input; PROJECT on infeasible input; `is_feasible()` confirmed on projected output |
| 2 | All Four Constraint Types | PASS | Linear + Quadratic + SOCP + PSD combined; 6 violations correctly identified on input `[12, 12, 12]`; projected to `[8.165, 8.165, 8.165]` |
| 3 | Fail-Closed (Theorem 2) | PASS | Contradictory constraints → REJECT; both violated constraint names returned |
| 4 | Hard Wall Dominance (Theorem 3) | PASS | REJECT with `solver_method='none'`; solver was never invoked |
| 5 | Enforcement Modes | PASS | All four mode/distance combinations produce correct decisions (`project`, `reject`, `project`, `reject`) |
| 6 | Passthrough & Idempotence (Theorem 9) | PASS | Two successive enforcements of a feasible point → both APPROVE, `distance=0.0` |
| 7 | Schema & Named Dimensions | PASS | Round-trip `vectorize → enforce → devectorize` returns exact input values |
| 8 | Budget Tracking | PASS | Budget depletes 250→170→100→40; rollback restores exactly 70 GPU-s (100→110); `abs(restored − 70.0) < 1e-6` |
| 9 | Audit Chain | PASS | 4 records (`decision`, `decision`, `decision`, `rollback`); SHA-256 chain intact; `prev_hash` matches predecessor `hash` at every link, `None` at genesis |
| 10 | Breaker State Machine | PASS | Scores 0.20→0.40 stay CLOSED; score 0.60 trips to THROTTLED at `trip_score=0.50`; score 0.80 stays THROTTLED |
| 11 | Global Default Policy | PASS | `lint_config()` → 0 issues; 30 schema fields, 81 named constraints, 3 budget specs |
| 12 | Policy Contract | PASS | Digest `afa6c306...` stable across `from_v5_config → to_dict → from_dict → to_dict` round-trip |
| 13 | Governor Lifecycle | PASS | 5 cycles gpu_util 0.20→0.80; cycles 1–4 APPROVE, cycle 5 REJECT (envelope ceiling hit); guarantee verified on every APPROVE/PROJECT via `is_feasible()` |
| 14 | Proof Checker | PASS | 3,732 / 3,732 structural and property assertions pass; exit 0 |

---

## Timing Breakdown

**Steps 1–13 combined: ~0.8 s.** The kernel itself is fast. All numerical work — convex projections, solver chain, budget arithmetic, SHA-256 hashing, breaker scoring, policy compilation, contract serialization, and governor lifecycle — completes in under one second.

**Step 14: ~32 s.** The proof checker dominates total runtime. It is a single-threaded Python script that executes 3,732 property assertions across the full proof structure (Axiom 1, Lemmas 1–3, Theorems 1–9, 2 Corollaries). This is expected: `verify_proof.py` is an offline verification tool, not a runtime component.

### Platform note — Windows console encoding

On Windows, `verify_proof.py` prints a Unicode checkmark (✓, U+2713) to stderr. The Windows default terminal encoding (cp1252) cannot represent that character, so running the script directly from a Windows shell exits with code 1 despite all 3,732 assertions passing. `hello_world.py` handles this by injecting `PYTHONIOENCODING=utf-8` into the subprocess environment, which is why Step 14 correctly reports exit 0. This is not a correctness issue — it is a Windows console encoding quirk. The script runs natively on Linux (the CI environment) with no workaround needed.

---

## Correctness Assessment

Every assertion in `hello_world.py` exercises a named theorem or invariant from `proof/PROOF.md`:

| Theorem | Exercised in step(s) | How |
|---------|----------------------|-----|
| Theorem 1 — enforcement guarantee | 1, 2, 13 | Explicit `is_feasible()` call on every APPROVE/PROJECT output |
| Theorem 2 — fail-closed | 3 | Contradictory region → REJECT confirmed |
| Theorem 3 — hard-wall pre-empts solver | 4 | `solver_method='none'` confirmed |
| Theorem 5 — budget monotonicity | 8 | Budget strictly non-increasing across three enforcements |
| Theorem 6 — rollback restoration | 8 | Rollback restores exact consumed amount to within 1e-6 |
| Theorem 8 — audit-chain integrity | 9 | SHA-256 chain verified link by link |
| Theorem 9 — passthrough / idempotence | 6 | Feasible input → APPROVE → re-enforce → APPROVE, distance=0.0 |
| 3,732 machine-checked assertions | 14 | `verify_proof.py` exits 0 — independent structural verification of the entire proof |

The script exercises both packages (`numerail` core kernel and `numerail_ext` survivability extension) in a single uninterrupted run. All 14 steps pass with zero failures.
