# Numerail v5.0.0 — Developer Documentation

Deterministic geometric enforcement for AI actuation safety.

---

## What Numerail does in one paragraph

Numerail sits between an AI system and the real world. When the AI proposes a numerical action — a token budget, a GPU lease, an API-call grant, a trade size — Numerail checks it against a set of mathematical constraints and returns one of three outcomes: **APPROVE** (the proposal already satisfies every constraint), **PROJECT** (the proposal was corrected to the nearest feasible point), or **REJECT** (the proposal was blocked). If the result is APPROVE or PROJECT, the output is mathematically guaranteed to satisfy every active constraint. This guarantee holds for all inputs, all models, and all solvers. The post-check is the trust boundary. The solver is untrusted.

---

## Quick-start: your first enforcement in 30 seconds

Install Numerail and its dependencies:

```bash
pip install -e .
```

Run the minimal example:

```python
import numpy as np
import numerail as nm

# Define a feasible region: all values between 0 and 1
region = nm.box_constraints([0, 0], [1, 1])

# An AI proposes a value outside the region
output = nm.enforce(np.array([1.5, 0.5]), region)

print(output.result.value)       # "project"
print(output.enforced_vector)    # [1.0, 0.5]
print(output.distance)           # 0.5

# THE GUARANTEE: the enforced output satisfies every constraint
assert region.is_feasible(output.enforced_vector)
```

That is Tier 1 — pure geometry with numpy arrays. No schema, no budgets, no audit trail. Just the mathematical guarantee.

The rest of this guide builds from here to a full production deployment using the AI resource governor as the running example.

---

## How the guarantee works

The guarantee is structural, not statistical. It does not depend on the AI model behaving well, on the solver producing correct results, or on any probability.

The `enforce()` function follows an eight-step pipeline:

1. **Post-check the input.** If the proposed vector already satisfies every constraint, return APPROVE.
2. **Check hard walls.** If any hard-wall constraint is violated, return REJECT immediately. No solver is invoked.
3. **Check reject mode.** If the enforcement mode is `"reject"`, return REJECT.
4. **Project via the solver chain.** Try box clamp (O(n), exact for axis-aligned boxes), then either Dykstra or SLSQP depending on constraint types, then the other as fallback. Each solver's output is independently post-checked.
5. **Post-check the solver output.** If no solver produced a verified feasible point, return REJECT.
6. **Check dimension policies.** If the correction changed a forbidden dimension, return REJECT.
7. **Check routing thresholds.** If the correction distance exceeds the hard-reject threshold, return REJECT.
8. **Check hybrid distance.** If in hybrid mode and the distance exceeds `max_distance`, return REJECT.

There is exactly one code path that returns APPROVE (gated by the feasibility check at step 1) and exactly one that returns PROJECT (reachable only after the solver's output passes the feasibility check at step 5). There are six code paths that return REJECT. Steps 6–8 can only convert a PROJECT into a REJECT — they cannot introduce an unverified approval.

A defense-in-depth check inside the output constructor uses an explicit `raise` (not a Python `assert`, so it cannot be stripped by `python -O`) to verify that every APPROVE or PROJECT output satisfies the feasibility check. This is redundant with the control flow — it exists to catch implementation bugs.

---

## Three tiers of access

Numerail provides three tiers. Each adds capability without changing the enforcement logic.

### Tier 1: Pure geometry

Call `enforce(vector, region)` with numpy arrays. No schema, no system, no budget, no audit. Just the mathematical guarantee.

```python
import numpy as np
from numerail import enforce, box_constraints, halfplane, combine_regions, EnforcementResult

# Box: each dimension between 0 and 100
region = box_constraints([0, 0, 0], [100, 100, 100])

# Add a joint constraint: sum of all dimensions ≤ 200
region = combine_regions(region, halfplane([1, 1, 1], 200.0, name="total_cap"))

# Enforce
output = enforce(np.array([90, 80, 70]), region)
print(output.result.value)        # "project" (sum=240 > 200)
print(output.enforced_vector)     # corrected to satisfy all constraints
assert region.is_feasible(output.enforced_vector)
```

### Tier 2: Named enforcement

Add a `Schema` for dict-to-vector translation and an `EnforcementConfig` for hard walls, forbidden dimensions, and routing.

```python
from numerail import enforce, box_constraints, Schema, EnforcementConfig, DimensionPolicy

schema = Schema(fields=["amount", "risk_score"])
region = box_constraints([0, 0], [1000, 1.0],
                         names=["max_amount", "max_risk", "min_amount", "min_risk"])
config = EnforcementConfig(
    mode="project",
    hard_wall_constraints=frozenset({"max_risk"}),
    dimension_policies={"risk_score": DimensionPolicy.PROJECTION_FORBIDDEN},
)

vec = schema.vectorize({"amount": 1200, "risk_score": 0.5})
output = enforce(vec, region, config, schema)
# amount exceeds max but risk_score is fine
# projection clips amount to 1000; risk_score is unchanged
```

### Tier 3: Full system

`NumerailSystem` integrates enforcement with budgets, rollback, audit, metrics, trusted context, and region versioning. `NumerailSystemLocal` wraps it in the full production code path with in-memory repositories.

This is what the AI resource governor example uses. The rest of this guide focuses on Tier 3.

---

## Building an AI resource governor: the complete walkthrough

This section walks through `examples/ai_resource_governor.py` step by step. By the end, you will understand every component of a production Numerail deployment.

### Step 1: Define the schema

The schema declares the fields that make up your action vector. Each field is a dimension in the enforcement geometry.

```python
fields = [
    "prompt_k",           # prompt-token ceiling, thousands
    "completion_k",       # completion-token ceiling, thousands
    "tool_calls",         # total tool-call allowance for this step
    "external_api_calls", # paid / side-effecting API calls
    "gpu_seconds",        # GPU lease for this step
    "parallel_workers",   # max concurrent subtasks
    "current_gpu_util",   # server-trusted GPU utilization [0,1]
    "current_api_util",   # server-trusted API utilization [0,1]
]
```

In the config dict, this becomes:

```python
"schema": {"fields": fields}
```

**Important:** Do not use schema normalizers unless you have a specific reason. Normalizers transform values before enforcement, which means the constraints must be written in normalized space. The AI resource governor example keeps all units raw and explicit to avoid this complexity.

### Step 2: Define linear constraints

Linear constraints are the workhorse. They express per-field bounds, structural relations, and headroom caps as rows of a matrix inequality `Ax ≤ b`. Each row has a name, which is how you reference it in budgets, hard walls, and error messages.

The AI resource governor defines 23 linear constraint rows:

**Per-step ceilings (8 rows).** Each field has an upper bound.

```
prompt_k         ≤ 64
completion_k     ≤ 16
tool_calls       ≤ 40
external_api_calls ≤ 20
gpu_seconds      ≤ 120
parallel_workers ≤ 16
current_gpu_util ≤ 1
current_api_util ≤ 1
```

**Zero floors (8 rows).** Every field must be non-negative.

**Structural relation (1 row).** External API calls cannot exceed tool calls:

```
external_api_calls - tool_calls ≤ 0
```

This means: every external API call must be backed by a tool call. The agent cannot claim it needs 4 tool calls but 8 external API calls.

**Headroom caps (2 rows).** The current utilization plus the requested resource, scaled, must leave at least 10% headroom:

```
current_gpu_util + gpu_seconds / 240 ≤ 0.90
current_api_util + external_api_calls / 40 ≤ 0.90
```

**Daily budget target rows (2 rows).** These are separate from the per-step maxes. They start at the daily budget initial value and shrink as resources are consumed:

```
gpu_seconds          ≤ 600   (remaining GPU day budget)
external_api_calls   ≤ 80    (remaining API day budget)
```

**Why budget rows must be separate from per-step max rows:** If a budget targets the same constraint row as a per-step max, the budget mechanism will overwrite the per-step max with the remaining budget. If the remaining budget (say, 400) exceeds the per-step max (120), the per-step max is effectively loosened. This is the "budget/cap collision" documented in `docs/SPECIFICATION.md`. Always use dedicated rows for budget targets.

In the config dict, linear constraints are specified as the `"polytope"` block:

```python
"polytope": {
    "A": [[1, 0, 0, 0, 0, 0, 0, 0],   # prompt_k ≤ 64
          [0, 1, 0, 0, 0, 0, 0, 0],   # completion_k ≤ 16
          ...],                          # 23 rows total
    "b": [64, 16, ...],                  # right-hand side
    "names": ["max_prompt_k", "max_completion_k", ...],
}
```

### Step 3: Define quadratic constraints

The quadratic constraint prevents the agent from maxing out every expensive dimension simultaneously. Even if each individual field is within its per-step ceiling, requesting the maximum of everything at once is too much.

The AI resource governor uses a normalized energy bound:

```
(prompt_k/64)² + (completion_k/16)² + (tool_calls/40)² +
(external_api_calls/20)² + (gpu_seconds/120)² + (parallel_workers/16)² ≤ 2.25
```

This means: if the agent requests 50% of every resource simultaneously (each ratio = 0.5), the left side is 6 × 0.25 = 1.5, which is under 2.25 — allowed. If the agent requests 100% of every resource (each ratio = 1.0), the left side is 6.0, which exceeds 2.25 — blocked.

The matrix Q is diagonal over the first 6 dimensions (the resource fields) with zeros on the last 2 (the trusted telemetry fields, which are not agent-controlled).

```python
"quadratic_constraints": [{
    "Q": [[1/64**2, 0, ...], ...],  # 8×8 diagonal matrix
    "a": [0, 0, 0, 0, 0, 0, 0, 0],
    "b": 2.25,
    "name": "resource_energy",
}]
```

Q must be positive semidefinite (PSD). Numerail verifies this at construction time via eigenvalue check and raises `ConstraintError` if it fails.

### Step 4: Define SOCP constraints

The second-order cone constraint models the combined burst footprint under live infrastructure conditions:

```
‖[gpu_util + gpu_sec/240, api_util + ext_api/40, workers/16]‖₂ ≤ 1.15
```

This says: the combined infrastructure load vector — current utilization plus what the agent is requesting — must have a Euclidean norm below 1.15. Under calm conditions (low utilization), the agent can make moderate requests. Under high utilization, even moderate requests become infeasible because the live telemetry values push the norm past the threshold.

This is what makes the agent's ability to act automatically reduce as the system heats up.

```python
"socp_constraints": [{
    "M": [[0, 0, 0, 0, 1/240, 0, 1, 0],   # gpu_util + gpu_sec/240
          [0, 0, 0, 1/40, 0, 0, 0, 1],     # api_util + ext_api/40
          [0, 0, 0, 0, 0, 1/16, 0, 0]],    # workers/16
    "q": [0, 0, 0],
    "c": [0, 0, 0, 0, 0, 0, 0, 0],
    "d": 1.15,
    "name": "burst_envelope",
}]
```

### Step 5: Define PSD constraints

The positive semidefinite (LMI) constraint couples GPU headroom, API headroom, and concurrency:

```
A(x) = [[1 - (gpu_util + gpu_sec/240),  0.35·workers/16],
         [0.35·workers/16,               1 - (api_util + ext_api/40)]]  ≽ 0
```

This 2×2 matrix must have all non-negative eigenvalues. When GPU headroom is tight (the (1,1) entry is small) and API headroom is tight (the (2,2) entry is small), the off-diagonal terms from `parallel_workers` can push the smaller eigenvalue below zero, making the constraint infeasible.

In plain English: when the system is already stressed, you cannot also fan out aggressively. This catches the pattern where uncontrolled agents become dangerous through parallel amplification rather than a single large action.

```python
"psd_constraints": [{
    "A0": [[1, 0], [0, 1]],
    "A_list": [
        [[0, 0], [0, 0]],           # prompt_k (no effect)
        [[0, 0], [0, 0]],           # completion_k (no effect)
        [[0, 0], [0, 0]],           # tool_calls (no effect)
        [[0, 0], [0, -1/40]],       # external_api_calls
        [[-1/240, 0], [0, 0]],      # gpu_seconds
        [[0, 0.35/16], [0.35/16, 0]],  # parallel_workers (off-diagonal)
        [[-1, 0], [0, 0]],          # current_gpu_util
        [[0, 0], [0, -1]],          # current_api_util
    ],
    "name": "coupled_headroom_psd",
}]
```

The `A_list` has one matrix per schema field (8 matrices for 8 fields). Fields that don't participate in the PSD constraint get zero matrices.

### Step 6: Declare trusted fields

Some fields should not come from the AI model. If enforcement depends on current GPU utilization and the agent supplies its own utilization value, the agent can spoof a low number to bypass headroom constraints.

Trusted fields solve this: the orchestrator (not the agent) injects the real values.

```python
"trusted_fields": ["current_gpu_util", "current_api_util"]
```

At enforcement time, the orchestrator provides `trusted_context`:

```python
result = local.enforce(
    proposed_action,
    trusted_context={"current_gpu_util": 0.74, "current_api_util": 0.55},
)
```

The agent's claimed utilization values are overwritten with the server-measured values before enforcement. Both the raw claims and the merged values are recorded in the feedback for auditability.

### Step 7: Configure enforcement behavior

```python
"enforcement": {
    "mode": "project",
    "dimension_policies": {
        "current_gpu_util": "forbidden",
        "current_api_util": "forbidden",
        "parallel_workers": "project_with_flag",
    },
    "routing_thresholds": {
        "silent": 0.05,
        "flagged": 1.5,
        "confirmation": 4.0,
        "hard_reject": 8.0,
    },
}
```

**Mode** controls what happens when the proposal violates constraints:

- `"project"` — correct to the nearest feasible point (default)
- `"reject"` — block all violations, no correction attempted
- `"hybrid"` — project if the correction distance is below `max_distance`, reject if above

**Dimension policies** control per-field projection behavior:

- `"forbidden"` — if projection would change this field, the result is REJECT. Used for trusted telemetry fields that the solver must never alter.
- `"project_with_flag"` — projection is allowed but the field name appears in `output.flagged_dimensions` for downstream attention.
- `"freely_projectable"` — the default. Projection can change this field silently.

**Routing thresholds** map correction distance to escalation tiers:

- Distance ≤ 0.05 → `SILENT_PROJECT` (auto-execute, log for metrics)
- Distance ≤ 1.5 → `FLAGGED_PROJECT` (execute, flag for post-hoc review)
- Distance ≤ 4.0 → `CONFIRMATION_REQUIRED` (hold for human approval)
- Distance > 8.0 → `HARD_REJECT` (converted to REJECT by Numerail)

Only `HARD_REJECT` is enforced by Numerail internally. The other tiers are metadata in the output — the orchestrator decides what to do with them.

### Step 8: Register budgets

Budgets make the feasible region shrink over time. Each approved action consumes part of the budget, and the corresponding constraint row's bound is reduced by the consumed amount.

```python
"budgets": [
    {
        "name": "gpu_day",
        "constraint_name": "remaining_gpu_day",
        "weight": {"gpu_seconds": 1.0},
        "initial": 600.0,
        "consumption_mode": "nonnegative",
    },
    {
        "name": "api_day",
        "constraint_name": "remaining_api_day",
        "weight": {"external_api_calls": 1.0},
        "initial": 80.0,
        "consumption_mode": "nonnegative",
    },
]
```

After the first enforcement approves `gpu_seconds=60`, the `remaining_gpu_day` constraint row's bound drops from 600 to 540. After the next enforcement approves `gpu_seconds=40`, it drops to 500. The agent's available GPU budget monotonically decreases.

**Consumption modes:**

- `"nonnegative"` (default) — consumption = max(0, weighted sum). Prevents the agent from "earning back" budget with negative values.
- `"abs"` — consumption = |weighted sum|.
- `"raw"` — consumption = weighted sum (allows negative consumption, for backward compatibility only).

**Weight maps:** The `"weight"` field can be a dict mapping field names to weights: `{"gpu_seconds": 1.0}` means consumption equals the enforced `gpu_seconds` value. For multi-field budgets, use multiple entries: `{"amount": 1.0, "fee": 1.0}` sums both fields.

### Step 9: Construct and run

```python
from numerail.local import NumerailSystemLocal

local = NumerailSystemLocal(config)

result = local.enforce(
    {"prompt_k": 32, "completion_k": 8, "tool_calls": 10,
     "external_api_calls": 5, "gpu_seconds": 60, "parallel_workers": 4,
     "current_gpu_util": 0.30, "current_api_util": 0.20},
    action_id="grant_1",
    trusted_context={"current_gpu_util": 0.30, "current_api_util": 0.20},
    execution_topic="runtime",
)

print(result["decision"])          # "approve"
print(result["enforced_values"])   # the granted envelope
print(result["audit_hash"])        # SHA-256 audit chain link
```

`NumerailSystemLocal` exercises the full production code path: policy parse, scope checks, trusted-context injection, budget handling, ledger rows, audit hash chain, outbox enqueue, and rollback. It uses in-memory implementations of every Protocol interface, so it behaves identically to a production deployment except that state lives in memory rather than a database.

### Step 10: Inspect the results

The result dict contains:

- `"decision"` — `"approve"`, `"project"`, or `"reject"`
- `"enforced_values"` — the granted envelope (dict of field name → value), or `None` if rejected
- `"feedback"` — structured feedback including violations, corrections, routing tier, budget status, and merged values
- `"audit_hash"` — the SHA-256 hash linking this decision to the audit chain
- `"action_id"` — the action identifier

For projected results, `feedback["corrections"]` shows per-field proposed vs. enforced values:

```python
feedback["corrections"]
# {"tool_calls": {"proposed": 4.0, "enforced": 6.0},
#  "external_api_calls": {"proposed": 8.0, "enforced": 6.0}}
```

For trusted-context overrides, `feedback["merged_values"]` shows what the enforcement actually used:

```python
feedback["raw_values"]["current_gpu_util"]      # 0.10 (agent claimed)
feedback["merged_values"]["current_gpu_util"]   # 0.65 (server injected)
```

### Step 11: Rollback

If downstream execution fails, rollback restores the budget:

```python
rb = local.rollback("grant_1")
print(rb.rolled_back)    # True
print(rb.audit_hash)     # audit chain records the rollback
```

Budget remaining is restored to its pre-enforcement value. The audit chain records both the original enforcement and the rollback. A second rollback of the same action_id raises `ValueError`.

### Step 12: Inspect infrastructure state

```python
local.budget_remaining    # {"gpu_day": 570.0, "api_day": 74.0}
local.outbox_events       # [{action_id, topic, payload, audit_hash}, ...]
local.audit_records       # [{type, hash, prev_hash, ...}, ...]
local.ledger              # {action_id: {raw_values, merged_values, ...}}
local.metrics             # {"enforcements": [...], "rollbacks": [...]}
```

Outbox events are enqueued only for approve and project decisions that have an `execution_topic`. Rejected actions never produce outbox events.

---

## Running the flagship example

```bash
python examples/ai_resource_governor.py
```

This runs five scenarios:

1. **Safe baseline grant** — all constraints satisfied, approved unchanged.
2. **Telemetry spoof attempt** — agent claims low utilization, server injects real (high) utilization, request becomes infeasible, rejected.
3. **Overlarge GPU burst (system hot)** — high utilization plus large GPU request violates multiple constraints (headroom, energy, burst envelope, coupled headroom), rejected.
4. **External API calls exceed tool calls** — structural violation (`external_api_calls > tool_calls`) projected to equality.
5. **Rollback** — demonstrates budget restoration after rollback.

---

## The policy configuration grammar

The complete config dict accepted by `NumerailSystemLocal(config)` and `NumerailSystem.from_config(config)`:

```python
{
    "policy_id": "string",              # optional identifier

    "schema": {
        "fields": ["field1", "field2", ...],
        "normalizers": {"field1": [lo, hi], ...},  # optional
        "defaults": {"field1": 0.0, ...},          # optional
    },

    "polytope": {
        "A": [[...], ...],             # m × n matrix (list of lists)
        "b": [...],                     # m-vector
        "names": ["row1", "row2", ...], # m names, must be unique
    },

    "quadratic_constraints": [{
        "Q": [[...], ...],             # n × n PSD matrix
        "a": [...],                     # n-vector
        "b": float,                     # scalar bound
        "name": "string",
    }],

    "socp_constraints": [{
        "M": [[...], ...],             # k × n matrix
        "q": [...],                     # k-vector
        "c": [...],                     # n-vector
        "d": float,                     # scalar bound
        "name": "string",
    }],

    "psd_constraints": [{
        "A0": [[...], ...],            # k × k symmetric matrix
        "A_list": [[[...], ...], ...], # n matrices, each k × k
        "name": "string",
    }],

    "trusted_fields": ["field1", "field2"],

    "enforcement": {
        "mode": "project",                          # or "reject" or "hybrid"
        "max_distance": float,                      # required for hybrid
        "hard_wall_constraints": ["name1", ...],
        "dimension_policies": {"field": "forbidden", ...},
        "routing_thresholds": {
            "silent": float, "flagged": float,
            "confirmation": float, "hard_reject": float,
        },
        "safety_margin": 1.0,                       # (0, 1]
        "solver_max_iter": 2000,
        "solver_tol": 1e-6,
        "dykstra_max_iter": 10000,
    },

    "budgets": [{
        "name": "string",
        "constraint_name": "linear_row_name",
        "weight": {"field": 1.0, ...},              # or scalar + dimension_name
        "initial": float,
        "consumption_mode": "nonnegative",           # or "abs" or "raw"
    }],
}
```

Use `lint_config(config)` to validate a config dict and get all issues at once (returns a list of strings, empty if valid). Use `PolicyParser().parse(config)` to validate strictly (raises on first error).

---

## Verification and testing

```bash
# Full test suite (138 tests)
pytest tests/ -v

# Guarantee certification suite only (45 tests, 7 categories)
pytest tests/test_guarantee.py -v

# Machine-verifiable proof checker (3,732 checks)
python proof/verify_proof.py

# AI resource governor tests (17 tests)
pytest tests/test_ai_resource_governor.py -v
```

The proof verifier imports from the local source tree deterministically (resolved from `__file__`, not from the current working directory) and reports the version dynamically from `numerail.__version__`.

---

## Exception hierarchy

All exceptions inherit from `NumerailError`:

| Exception | When it fires |
|---|---|
| `ValidationError` | NaN, Inf, dimension mismatch, invalid mode, bad margin |
| `ConstraintError` | Non-PSD matrix, zero weights, shape mismatch, duplicate names |
| `InfeasibleRegionError` | Available for user code; raised when a region is confirmed empty |
| `SolverError` | SLSQP convergence failure (informational; enforce returns REJECT) |
| `SchemaError` | Missing field, duplicate field, unknown dimension policy key |
| `ResolutionError` | Constraint name not found, hard-wall name unknown, budget target unknown |
| `AuthorizationError` | Caller lacks required scope (production layer only) |

---

## Module structure

```
src/numerail/
    __init__.py    — public API surface (re-exports everything)
    engine.py      — mathematical kernel: constraints, region, projection,
                     enforcement, schema, budgets, audit, metrics, system
    parser.py      — strict policy parser + lint_config
    service.py     — production runtime service with transactional flow
    local.py       — in-memory local mode (exercises production code path)
    protocols.py   — typed Protocol interfaces for production repositories
    errors.py      — production-layer exceptions

tests/
    test_engine.py               — engine-level tests (37 tests)
    test_guarantee.py            — guarantee certification suite (45 tests)
    test_parser.py               — parser and linter tests (14 tests)
    test_service.py              — production path tests (25 tests)
    test_ai_resource_governor.py — flagship example tests (17 tests)

examples/
    quickstart.py                — minimal 2D box example
    ai_resource_governor.py      — flagship: full AI governance demo

proof/
    PROOF.md                     — mathematical proof (9 theorems)
    verify_proof.py              — machine-verifiable proof checker (3,732 checks)
```

The engine is a single file by design. The entire trust boundary — from `ConvexConstraint.evaluate()` through `FeasibleRegion.is_feasible()` through the `_out()` invariant to the eight return paths of `enforce()` — is auditable in one top-to-bottom read without opening another file.

---

## Production deployment: the service layer

For production deployments with persistent state, use `NumerailRuntimeService` directly with your own repository implementations.

The service layer wraps the unchanged engine in transactional infrastructure: scoped authorization, protocol-based repositories, strict policy parsing, audit trail, metrics collection, and outbox publishing.

The transactional flow for each enforcement call:

1. Authorize the caller (check scopes)
2. Load the active policy version
3. Lock budget rows (`SELECT ... FOR UPDATE`)
4. Parse and validate the policy
5. Build the engine from the policy config
6. Merge trusted context
7. Enforce
8. Persist budget deltas
9. Write audit record
10. Write ledger entry
11. Update metrics
12. Enqueue outbox event (if approved/projected and topic is set)
13. Commit

If any step fails, the entire transaction rolls back atomically.

You implement the repository interfaces defined in `numerail.protocols`:

- `TransactionManager` — context manager for database transactions
- `AuthorizationService` — scope checking
- `PolicyRepository` — load active policy config
- `RuntimeRepository` — lock and mutate budget state
- `LedgerRepository` — action decision records
- `AuditRepository` — hash-linked audit chain
- `MetricsRepository` — enforcement metrics
- `OutboxRepository` — downstream execution events

`NumerailSystemLocal` provides in-memory implementations of all seven, which is why it is the right tool for development and testing.

---

## Formal properties

Numerail's behavior is characterized by nine theorems, proved in `proof/PROOF.md` and independently verified by `proof/verify_proof.py`:

1. **Soundness.** APPROVE or PROJECT implies the output satisfies every constraint.
2. **Fail-closed.** If all solvers fail, the result is REJECT.
3. **Hard-wall dominance.** A hard-wall violation produces REJECT with no solver invoked.
4. **Forbidden-dimension safety.** If correction would change a forbidden dimension, REJECT.
5. **Budget monotonicity.** Under nonnegative consumption, the feasible region can only shrink.
6. **Rollback restoration.** Rollback restores the exact per-action budget delta.
7. **Monotone self-limits.** In reject mode with version control, the agent cannot re-escalate.
8. **Audit integrity.** Under SHA-256 collision resistance, any audit tampering is detectable.
9. **Passthrough and idempotence.** A feasible input is approved unchanged. Enforcing an already-enforced output produces APPROVE.

---

## What the guarantee does not cover

The guarantee proves that the output satisfies every stated constraint. It does not prove:

- **That the constraints are correct.** A GPU ceiling of 1,200 seconds when the intent was 120 will be faithfully enforced at 1,200. The specification is the deployer's responsibility.

- **That the correction direction is intended.** Projection finds the geometrically nearest feasible point, which may preserve the aspect of the proposal that was dangerous. This is why routing thresholds, forbidden dimensions, hard walls, and hybrid mode exist — they control the projection gap.

- **That external context is accurate.** If trusted field values are wrong, enforcement is correct for the wrong merged action.

The guarantee's value is that it converts an unbounded risk (the AI could output anything) into a bounded, auditable, formally characterized one (the AI's output satisfies every stated constraint, and the residual risk is entirely in the constraints themselves).

---

*Numerail governs the execution envelope of an AI system: the model may request compute and tool authority, but the runtime only receives a grant that satisfies declared policy. The guarantee applies to conformance with the encoded feasible region, not to the correctness or completeness of the policy specification itself.*
