# Changelog

All notable changes to Numerail are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Both packages use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Lean 4 machine-checked formalization of the enforcement guarantee (`proof/Guarantee.lean`).** Formalizes the same enforcement guarantee in Lean 4 with Mathlib. 12 proofs verified by the Lean kernel (9 public, 3 private supporting), 0 `sorry`. Public: `is_feasible_correct`, `enforcement_soundness` (Theorem 1), `enforcement_soundness_per_constraint`, `fail_closed` (Theorem 2), `hard_wall_dominance` (Theorem 3), `passthrough`, `idempotence` (Theorem 9), `budget_monotonicity` (Theorem 5 fragment), `numerail_guarantee`. Private: `approve_sound`, `project_sound`, `reject_absurd`. Three axioms identical in intent to the Rocq formalization: abstract `Vec` type, `solver` black box + `project_postcheck`, and `operational_filters_pass`. The Rocq and Lean 4 formalizations are fully independent — different proof assistants, different proof kernels, no shared code — and both verify identical theorem statements. Requires Lean 4 with Mathlib. See `proof/PROOF.md § Lean 4 Machine-Checked Formalization` for full details.

- **Rocq/Coq machine-checked formalization of the enforcement guarantee (`proof/Guarantee.v`).** Formalizes Axiom 1, Lemma 1, Lemmas 2–3, and Theorem 1 (enforcement soundness) in Rocq/Coq using only the standard library. 11 proofs, 0 `Admitted`. Proves `enforcement_soundness` (Theorem 1), `enforcement_soundness_per_constraint` (per-constraint corollary), `fail_closed` (Theorem 2), `hard_wall_dominance` (Theorem 3), `passthrough` and `idempotence` (Theorem 9), `budget_monotonicity` (Theorem 5 structural fragment), and supporting lemmas. Three axioms/parameters: abstract `vec` type (floating-point not modelled — see PROOF.md abstraction gap note), `solver` black box + `project_postcheck` axiom formalising Lemma 2, and `operational_filters_pass` predicate covering the operational filter paths. Compiles on Rocq 9.0 / Coq 8.18+. See `proof/PROOF.md § Rocq Machine-Checked Formalization` for full details.

- **`test_mathematical_guarantees.py` — 99-test mathematical guarantee analysis suite.** One test per proof claim across 21 sections (A–U), covering Axiom 1, Lemmas 1–3, Theorems 1–9, both Corollaries, quantitative precision, projection optimality, convexity, routing thresholds, trusted context, schema, and stress tests. Includes a plain-language preface accessible to non-technical readers. Brings core test total to 253.

- **New examples: `autonomous_agent_governor.py`, `rest_api_server.py`, `rest_api_client.py`.** `autonomous_agent_governor.py` — 20-step simulation demonstrating breaker transitions, budget depletion, and rollback. `rest_api_server.py` — FastAPI server wrapping `NumerailSystemLocal` with three endpoints (`POST /enforce`, `POST /rollback/{action_id}`, `GET /budgets`) and trusted-context injection via `X-Trusted-Context` header. `rest_api_client.py` — stdlib-only client (urllib + json) exercising all three endpoints including APPROVE, PROJECT, REJECT, rollback, 404, and 409 cases.

### Fixed

- **Stale documentation across four files (QA audit).** (1) `README.md` line 56: "45 certification tests" corrected to 46. (2) `docs/GUARANTEE.md` lines 36, 92: "45 certification tests" corrected to 46. (3) `docs/GUARANTEE.md` line 31: bare `assert` code snippet replaced with the actual `raise AssertionError(...)` block, with note that it cannot be stripped by `python -O`. (4) `docs/GUARANTEE.md` line 50: stale claim that the APPROVE path uses the hardcoded default tolerance updated to reflect that both APPROVE and PROJECT paths use `cfg.solver_tol` uniformly after the R1 gate fix.

- **`engine.py` — tolerance mismatch between R1 gate and `_out` defense-in-depth (`engine.py` line 1266).** The APPROVE path checked `effective.is_feasible(x)` using the hardcoded default tolerance `1e-6`, while the defense-in-depth assertion in `_out()` checked `effective.is_feasible(enforced, cfg.solver_tol)`. When a caller configured `solver_tol` tighter than `1e-6`, vectors with violations in `(solver_tol, 1e-6]` passed the R1 gate but then raised `AssertionError` inside `_out()` — a crash on valid inputs. Fixed by passing `cfg.solver_tol` explicitly to the R1 gate: `effective.is_feasible(x, cfg.solver_tol)`. Behavior is identical at the default `solver_tol=1e-6`. The fix makes the APPROVE gate and the defense-in-depth assertion use the same tolerance under all configurations. Safety impact: none — the system was failing closed; no unsound output was ever returned. Also updated the structural pattern check in `proof/verify_proof.py` and `tests/test_guarantee.py` to accept the tol-explicit call form, and added `TestTolerance::test_approve_respects_solver_tol` as a regression test.

---

## numerail 5.0.0 — 2025

### Added

**Enforcement kernel (`engine.py`)**
- `enforce()`: eight-step enforcement pipeline with formally proved soundness guarantee (Theorem 1). Approve path gated by `is_feasible(x)`. Project path gated by `proj.postcheck_passed`. Six independent reject paths. Steps 6–8 satisfy the operational policy monotonicity lemma — they can only convert project to reject, never introduce unverified approvals.
- Defense-in-depth invariant in the emit path: explicit `raise AssertionError` (not Python `assert`) guards every APPROVE and PROJECT return. Cannot be stripped by `python -O`.
- `FeasibleRegion`: intersection of convex constraints. `is_feasible()` is the combined post-check and the system trust boundary. Composition via `combine()` and `add_constraint()`. Copy-on-modify bound updates for budget integration. Named constraint index with duplicate detection.
- `LinearConstraints` (`Ax ≤ b`): componentwise checker, iterative half-space `project_hint`, `with_bound()` and `with_safety_margin()` copy-on-modify operators, per-row diagnostics (`row_violations`, `row_slack`, `row_bindings`).
- `QuadraticConstraint` (`x'Qx + a'x ≤ b`): PSD verification at construction via eigenvalue check. KKT bisection `project_hint` with 100-iteration binary search on dual variable λ.
- `SOCPConstraint` (`‖Mx + q‖ ≤ c'x + d`): second-order cone constraint. Pre-image of the Lorentz cone under an affine map.
- `PSDConstraint` (`A₀ + Σ xᵢAᵢ ≽ 0`): linear matrix inequality. `evaluate()` returns −λ_min. O(k³) eigendecomposition post-check.
- Solver chain: box clamp (O(n), exact for axis-aligned boxes) → SLSQP (general convex) → Dykstra (guaranteed convergence for convex intersections). Routing: linear-only regions use Dykstra first; mixed/nonlinear use SLSQP first. Every solver's output independently post-checked.
- `Schema`: field-to-index mapping with optional affine normalizers and defaults. `vectorize()` / `devectorize()`. Validates dimension policy keys at construction and enforcement time.
- `BudgetSpec` and `BudgetTracker`: name-based budget tracking with rollback support. Three consumption modes: `nonnegative`, `abs`, `raw`. Weight maps for multi-field budgets. Thread-safe with per-action delta store. `set_consumed()` for service-layer synchronisation.
- `AuditChain`: append-only SHA-256 hash-linked log. Bounded memory with correct chain verification after eviction. `verify()` walks from genesis, recomputing every hash.
- `MetricsCollector`: thread-safe approve/project/reject rates, correction distance distribution (mean, p99, max), solver distribution, top-violated and top-binding constraint names.
- `RegionVersionStore`: immutable versioned region history. Rollback to any retained version.
- `NumerailSystem`: production integration class. Thread-safe `enforce()` with budget application, audit appending, metrics recording, and trusted context merge. `rollback()` restores exact per-action budget delta and syncs region. `from_config()` factory accepts JSON-compatible dict.
- `merge_trusted_context()`: standalone trusted context merge function.
- `synthesize_feedback()`: structured agent-facing enforcement feedback.
- Builder functions: `box_constraints()`, `halfplane()`, `combine_regions()`, `ellipsoid()`.
- Analysis utilities: `check_feasibility()` (exact LP for linear, best-effort projection for mixed), `chebyshev_center()` (largest inscribed ball for linear regions).
- `EnforcementConfig`: frozen dataclass. Mode (`project`, `reject`, `hybrid`), routing thresholds, dimension policies, hard wall constraints, safety margin, solver parameters.
- `RoutingThresholds`: distance-to-escalation tier mapping (SILENT_PROJECT, FLAGGED_PROJECT, CONFIRMATION_REQUIRED, HARD_REJECT). HARD_REJECT is enforced internally; other tiers are feedback metadata.
- `DimensionPolicy`: FREELY_PROJECTABLE, PROJECT_WITH_FLAG, PROJECTION_FORBIDDEN.
- Backward compatibility: `Polytope` wrapper, `ActionSchema`, `GCESystem`, `GCEError` aliases.

**Production layer**
- `PolicyParser`: strict policy parser with fail-fast validation. Accepts both `action_schema` and `schema` keys for backward compatibility. Validates dimension policy keys, hard wall constraint names, budget targets, and trusted field names.
- `lint_config()`: collect-all-issues validator. Detects budget/cap collisions (initial value exceeds target row bound), unknown constraint names, duplicate field names, missing required fields, and invalid mode configurations.
- `NumerailRuntimeService`: transactional enforcement service. Authorise → lock → parse → build → enforce → persist → commit. Computes per-action budget delta by diffing consumed totals before and after enforcement.
- `NumerailSystemLocal`: in-memory local mode exercising the full production code path. All seven Protocol interfaces implemented with in-memory state.
- Typed Protocol interfaces for all repository dependencies: `TransactionManager`, `AuthorizationService`, `PolicyRepository`, `RuntimeRepository`, `LedgerRepository`, `AuditRepository`, `MetricsRepository`, `OutboxRepository`.
- Scope enforcement: `enforce` scope required for all enforcement; `trusted:inject` scope required to supply trusted context; `rollback` scope required for rollback.

**Formal verification**
- `proof/PROOF.md`: mathematical proof of nine theorems. Axiom 1 (checker correctness for all four constraint types), Lemma 1 (combined checker correctness), Lemma 2 (project post-check), Lemma 3 (emit path invariant), Theorem 1 (enforcement soundness), Theorem 2 (fail-closed rejection), Theorem 3 (hard-wall dominance), Theorem 4 (forbidden-dimension safety), Theorem 5 (budget monotonicity), Theorem 6 (rollback restoration), Theorems 7–9 (monotone self-limits, audit integrity, passthrough and idempotence).
- `proof/verify_proof.py`: 3,732 machine-verifiable structural and property checks against `engine.py` source. Zero admitted lemmas.

**Test suite**
- `test_engine.py`: 37 tests covering all constraint types, enforcement modes, schema, budgets, audit chain, solver routing.
- `test_guarantee.py`: 46 certification tests across 7 categories: structural verification, formal property tests (all 9 theorems), constraint type coverage, adversarial probes, randomised stress tests, enforcement mode coverage, tolerance precision. Includes `test_approve_respects_solver_tol` regression test for the R1 gate tolerance fix.
- `test_mathematical_guarantees.py`: 99 guarantee analysis tests (one per proof claim) across 21 sections (A–U), covering Axiom 1, Lemmas 1–3, Theorems 1–9, both Corollaries, quantitative precision, projection optimality, convexity, routing thresholds, trusted context, schema, and stress tests. Includes a plain-language preface accessible to non-technical readers.
- `test_parser.py`: 14 tests covering config grammar, budget collision detection, hard-wall validation.
- `test_service.py`: 25 tests covering the full transactional flow, scope enforcement, weight-map budgets.
- `test_ai_resource_governor.py`: 17 tests for the flagship AI governance example.
- `test_ai_circuit_breaker.py`: 15 tests for the control-plane reserve pattern.

**Documentation**
- `README.md`: guarantee statement, quickstart, constraint types, architecture overview, verification commands, examples index.
- `docs/DEVELOPER_GUIDE.md`: complete walkthrough from schema definition through all four constraint types to production deployment. Step-by-step AI resource governor example.
- `docs/GUARANTEE.md`: what the guarantee is, why it holds, what it depends on, what it does not cover.
- `docs/SPECIFICATION.md`: the specification problem — the four phases of constraint engineering, the five failure types, validation tools.
- `docs/DEPLOYMENT.md`: four deployment roles, transactional enforcement flow, trusted context injection, budget lifecycle, outbox pattern, audit and evidence, operational metrics, assurance conditions, control-plane reserve pattern, defense-in-depth invariant.
- `docs/REFERENCE.md`: API reference, config grammar, constraint types, solver chain, all nine formal properties.

---

## numerail-ext 0.4.0 — 2025

### Added

**Breaker state machine (`breaker.py`, `types.py`)**
- `BreakerStateMachine`: hysteretic supervisor. Weighted overload score `s = 0.30·u_gpu + 0.25·u_api + 0.20·u_db + 0.10·u_queue + 0.15·min(1, e/10)`. Five modes: CLOSED (α=1.00), THROTTLED (α≈0.50), HALF_OPEN (α≈0.25), SAFE_STOP (α≈0.05, latched), OPEN (α=0.00, all authority suspended).
- Hysteretic transition logic: CLOSED → THROTTLED at trip_score; THROTTLED → HALF_OPEN at reset_score; HALF_OPEN → CLOSED at reset_score; HALF_OPEN → THROTTLED on re-trip; any mode → SAFE_STOP at safe_stop_score. SAFE_STOP is a latched state — `manual_reset()` required, denied if overload score exceeds reset_score at call time.
- Thread-safe via internal lock on all mode reads and transitions.
- `BreakerThresholds`: frozen dataclass. Validates `reset_score ≤ trip_score ≤ safe_stop_score` at construction.
- `TelemetrySnapshot`: frozen dataclass. 14 fields partitioned into live utilisation, protected reserves, and disturbance margins. All server-authoritative.
- `WorkloadRequest`: frozen dataclass. 13 AI-proposed workload fields. `as_action_dict()` serialises to the V5 schema format.
- `BreakerDecision`, `TransitionEnvelope`, `ExecutableGrant`, `ExecutionReceipt`, `GovernedStep`: typed data structures for the full enforce/commit lifecycle.
- `TransitionModel`, `ReservationManager`, `Digestor`, `NumerailBackend`: Protocol interfaces for the governor's collaborators.

**Transition model (`transition_model.py`)**
- `IncidentCommanderTransitionModel`: concrete `TransitionModel` implementation. Synthesises a `TransitionEnvelope` from (mode, telemetry, budgets) applying mode scale factors, available physical headroom (capacity minus utilisation minus reserve minus disturbance margin), and remaining shift budget ceilings. Envelope caps are strictly monotone non-increasing through the mode hierarchy (CLOSED ≥ THROTTLED ≥ HALF_OPEN ≥ SAFE_STOP ≥ OPEN=0). `next_state_safe()` validates post-execution state.
- Three-layer freshness architecture: L1 V5 engine constraints (state_version ≥ min_required, observed_at_ns ≤ expires_at_ns); L2 governor clock check (time_ns() ≤ snapshot.observed_at_ns + freshness_ns); L3 reservation manager acquire (validates token at commit time).
- Reserve-aware headroom constraints include three terms: current utilisation, AI-proposed consumption, and control-plane reserve plus disturbance margin. Trusted reserve fields are FORBIDDEN — the solver cannot alter them.

**Policy builder (`policy_builder.py`)**
- `build_v5_policy_from_envelope()`: compiles a `TransitionEnvelope` into a complete V5-compatible config dict. Generates the full 30-field schema, ~78 linear constraints (box bounds, structural invariants, freshness conditions, reserve-aware headroom, budget target rows), 1 quadratic (workload energy bound), 1 SOCP (joint infrastructure burst envelope, ‖Mx‖ ≤ 1.75), 1 PSD (coupled cross-channel headroom, 3×3 matrix LMI). Trusted fields declared FORBIDDEN. Budget specs with canonical weight maps.

**Global default policy pack (`global_default.py`)**
- `build_global_default()`: produces a complete V5-compatible config dict with conservative defaults. 30 fields (13 workload, 4 sequencing/freshness, 5 live state, 4 protected reserves, 4 disturbance margins). Design invariant: the default is denial.
- Module 1 (action authority bounds): per-step caps on all 13 workload fields. Non-negativity floors on all 30 fields. Upper bounds on all 17 trusted fields.
- Module 2 (structural invariants): 7 prerequisite relations encoding operational knowledge — external API calls require internal tool calls; cloud mutations required before feature flag changes, rollbacks, and traffic shifts; external API budget covers communications channels; worker scale-up proportional to current parallelism.
- Module 3 (freshness and sequencing): state_version ≥ min_required_state_version; observed_at_ns ≤ expires_at_ns.
- Module 4 (reserve-aware headroom): GPU, API, DB, and queue headroom constraints including current utilisation, proposed consumption, control-plane reserve, and disturbance margin.
- Module 5 (reserve bounds): trusted field floors and caps for all four reserve and four margin fields.
- Module 6 (advanced constraints): quadratic workload energy bound, SOCP joint burst envelope, PSD coupled cross-channel headroom. SOCP catches coordinated burst patterns that pass individual headroom checks. PSD catches parallel amplification when multiple headroom channels are simultaneously tight.
- Module 7 (shift budgets): three per-shift budgets — `gpu_shift` (3,600 GPU-seconds), `external_api_shift` (500 calls), `mutation_shift` (100 mutations) — with canonical weight maps and nonnegative consumption mode.
- Dimension policies: all 17 trusted fields FORBIDDEN; three sensitive workload fields PROJECT_WITH_FLAG.
- Routing thresholds: four-tier escalation (silent → flagged → confirmation_required → hard_reject).
- Constraint modules tagged to EU AI Act Articles 9, 12, 14, and 15.

**Governor (`governor.py`)**
- `StateTransitionGovernor`: supervisory enforcement governor. 12-step enforce/commit lifecycle. Step 1: L2 freshness check (force SAFE_STOP if stale). Step 2: breaker update (compute overload score, transition mode). Step 3: OPEN fast-path (return REJECT without invoking V5 engine). Steps 4–6: synthesise envelope, compile policy, load into backend. Step 7: acquire reservation. Step 8: merge trusted context, invoke V5 enforcement. Step 9: issue HMAC-signed `ExecutableGrant` on approve/project. Step 10: validate receipt (action_id + state_version + payload_digest + expiry). Step 11: `next_state_safe()` check (SAFE_STOP + rollback if unsafe). Step 12: commit reservation.
- Policy rebuilt every cycle — envelope ceilings depend on current telemetry, not just breaker mode, so a cached policy from the same mode but different utilisation would be stale.
- `bootstrap_budgets` parameter: initial budget state for the first enforcement cycle before the backend has loaded any config. Governor raises `RuntimeError` rather than silently compiling a zero-budget policy when no budget state exists.
- `manual_reset()`: delegates to `BreakerStateMachine.reset()`. Clears SAFE_STOP only when overload score is at or below reset threshold.

**Policy contract (`contract.py`)**
- `NumerailPolicyContract`: content-addressable, chain-linked, V5-native policy contract. Five top-level sections: header (provenance, chain linkage), geometry (constraint definitions), trust (trusted fields, dimension policies), enforcement (mode, routing, solver config), budgets.
- Content-addressable: digest = SHA-256(canonical_JSON(digestable_dict)). Canonical serialisation matches V5's `_deterministic_json` — sorted keys, no whitespace, numpy type coercions. Two contracts with identical content produce identical digests regardless of construction time.
- Chain-linked: `header.previous_digest` points to predecessor's digest. Genesis has `previous_digest=None`. `verify_chain()` walks the sequence, recomputes every digest, verifies every link.
- Tamper detection: `from_dict()`, `from_json()`, `from_bytes()` all recompute the digest and raise `ValueError` on mismatch. Every deserialisation path is a tamper-detection boundary.
- `v5_config` property: extracts a complete `NumerailSystem.from_config()`-compatible dict. Merges dimension policies into enforcement section, promotes trusted fields to top-level. Header and digest excluded from V5 config.
- Wire format: `to_bytes()` / `from_bytes()` via UTF-8 canonical JSON. Independently verifiable using only `hashlib` and `json` from Python's stdlib — no Numerail installation required.
- `from_v5_config()` factory: wraps any V5-compatible config dict in a contract with provenance metadata. Accepts configs from `build_global_default()`, `build_v5_policy_from_envelope()`, or hand-written configurations. Chaining via `previous_digest` parameter.
- `ContractHeader`, `ContractGeometry`, `ContractTrust`, `ContractEnforcement`, `ContractBudget`, `ContractActivation`: typed frozen dataclass sections.
- `summary()`: human-readable summary including digest prefix, policy identity, constraint summary, trust counts, and chain position.

**Supporting modules**
- `validation.py`: `validate_receipt_against_grant()` — post-execution trust boundary. Verifies action_id identity, state_version consistency, payload_digest tamper detection, and execution within freshness window. Raises `ReceiptValidationError` on any mismatch.
- `local_backend.py`: `LocalNumerailBackend` — in-memory `NumerailBackend` Protocol implementation. Wraps `NumerailSystemLocal`. Rebuilds the local system on each `set_active_config()` call so that envelope-derived ceilings are always current.

**Test suite**
- `test_breaker_mode.py`: 87 tests across 10 test classes: `TestBreakerStateMachine` (15), `TestTransitionModel` (9), `TestPolicyBuilderV5` (8), `TestValidation` (5), `TestLocalBackend` (3), `TestGovernor` (13), `TestGuaranteeFuzz` (20), `TestTypes` (5), `TestModeBoundary` (6), plus additional type and boundary tests.
- `test_contract.py`: 120 tests across 13 test classes covering all documented properties of `NumerailPolicyContract`: `TestContentAddressableIdentity` (8), `TestDigestComputation` (10), `TestChainLinkage` (8), `TestV5ConfigExtraction` (10), `TestWireFormat` (15), `TestPortableVerification` (3), `TestTrustPartition` (7), `TestBudgetSpecifications` (9), `TestGeometryIntrospection` (10), `TestEnforcementConfig` (9), `TestBreakerSuiteCompatibility` (5), `TestIntrospection` (14), `TestFactory` (9).

### Fixed

- **`contract.py` — `ContractGeometry.to_dict()` shallow reference bug.** `to_dict()` returned a dict whose values were direct references to the same internal list objects (e.g. `polytope['b']`) stored in the frozen dataclass. `@dataclass(frozen=True)` prevents attribute reassignment but does not protect the contents of mutable objects those attributes reference. Any caller who modified the returned dict would silently corrupt the contract's internal state, causing subsequent `verify_digest()` calls to return `False` and `from_bytes()` to raise `ValueError`. Fixed by applying `copy.deepcopy()` to all five geometry fields in `to_dict()`, ensuring the returned dict is fully independent of the contract's internal state. Detected by `TestWireFormat::test_from_dict_raises_on_tampered_constraint` corrupting the module-scoped `genesis` fixture shared across all 120 contract tests.

---

[Unreleased]: https://github.com/Numerail/Numerail/compare/v5.0.0...HEAD
[5.0.0]: https://github.com/Numerail/Numerail/releases/tag/v5.0.0
