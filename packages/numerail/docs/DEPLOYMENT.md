# Deployment Guide

---

## The deployment architecture

Numerail is designed as a three-layer system. The engine is the mathematical core. The policy layer compiles JSON configuration into typed constraint geometry. The runtime layer wraps the engine in transactional persistence, authentication, trusted context injection, and outbox delivery.

The trust boundary is in the engine, not the API. The API is a serialization and transaction layer around the unchanged enforcement logic. This means the guarantee is portable: the same engine, with the same proofs, can be wrapped in a different API, a different transport, or embedded directly in an application.

## Four deployment roles

**Agent.** Proposes actions. Has only the `enforce` scope. Cannot create, activate, or modify policies, trusted context declarations, or budget configurations. The agent's API key is scoped to enforcement only.

**Orchestrator.** Enriches actions with server-authoritative trusted context, calls the enforcement endpoint, handles downstream execution and rollback. Has `enforce`, `rollback`, and `trusted:inject` scopes. The orchestrator is the bridge between the agent and the enforcement system.

**Policy authority.** Defines and versions constraint geometry. Has `policy:write` and `policy:version` scopes. Policy authorities are domain experts (compliance officers, risk managers, platform reliability owners, infrastructure governance owners) who translate governance intent into constraint specifications.

**Auditor.** Reviews enforcement history and metrics. Has `audit:read` and `metrics:read` scopes. Auditors verify that the system is operating within its governance mandate by examining the audit chain, version history, and operational metrics.

## Transactional enforcement flow

A single enforcement call in the deployed system should run as one database transaction:

1. Load active policy version.
2. Lock runtime head and budget rows with `SELECT ... FOR UPDATE`.
3. Merge trusted context (server-supplied values overwrite agent-supplied values for declared trusted fields).
4. Call the cached engine — the unchanged `enforce()` function.
5. Compute and persist budget deltas.
6. Insert action ledger record (containing both the agent's raw proposal and the merged values used for enforcement).
7. Append audit record to the hash-linked chain.
8. Update metrics.
9. Optionally enqueue an outbox event for downstream execution.
10. Commit.

If any step fails, the entire transaction rolls back atomically. There is no state where the budget was consumed but the audit record wasn't written, or where the decision was persisted but the outbox event wasn't enqueued.

This transactional atomicity is not part of the geometric guarantee (Theorem 1). It is an operational property that ensures the deployed system maintains coherence between enforcement state and execution state. The paper's addendum names this as condition C2 (transactional integrity) in the compositional assurance argument.

## Trusted context injection

Some fields should not come from the model at all. If the enforcement depends on a fraud risk score and the agent provides its own risk score, the agent can spoof a low value and bypass the constraint. The trusted context mechanism prevents this by architecture.

The orchestrator (not the agent) provides the authoritative value via the `trusted_context` parameter. The engine's `merge_trusted_context()` function overwrites only fields that are declared as trusted in the policy. The merge happens before vectorization, so enforcement sees the server's value, not the agent's claim.

Both values — the agent's raw proposal and the server's injected values — are recorded in the enforcement feedback for full auditability. The action ledger preserves the agent's lie alongside the server's correction.

```python
system.set_trusted_fields(frozenset({"risk_score"}))

result = system.enforce(
    {"amount": 25, "risk_score": 0.0},       # agent claims risk 0.0
    trusted_context={"risk_score": 0.85},      # server knows risk is 0.85
)
# Enforcement uses risk_score=0.85
# feedback["raw_values"]["risk_score"] == 0.0  (agent's claim, recorded)
# feedback["merged_values"]["risk_score"] == 0.85  (server's value, used)
```

Trusted context is double-gated: only fields listed in the policy's `trusted_fields` can be overwritten, and only callers with the `trusted:inject` scope can supply trusted context. If the caller provides trusted_context without the required scope, the service raises `AuthorizationError`. This is deliberate — explicit failure is safer and more auditable than silent discard.

For AI-system governance, trusted context is the mechanism that prevents the agent from lowering its own admissibility barrier. For example, if enforcement depends on current GPU utilization:

```python
# Agent claims GPU utilization is 0.10 (low — would allow a large lease)
# Orchestrator injects server-measured utilization of 0.74 (high — constrains the lease)
result = local.enforce(
    {"gpu_seconds": 80, "current_gpu_util": 0.10, ...},
    trusted_context={"current_gpu_util": 0.74},
)
# Enforcement uses current_gpu_util=0.74
# The large GPU lease becomes infeasible under real utilization → REJECT
```

### Automatic trusted context injection via TrustedContextProvider

The service-layer mechanism described above requires the orchestrator to supply
trusted values explicitly on each call.  For fields whose values are always
server-authoritative — especially the wall-clock time — you can eliminate the
call-site burden by attaching a `TrustedContextProvider` to `NumerailSystemLocal`
at construction.  The provider is called automatically on every `enforce()` call.

```python
from numerail.local import NumerailSystemLocal, DefaultTimeProvider

# DefaultTimeProvider is the default — inject current_time_ms on every call.
local = NumerailSystemLocal(config)

# Custom provider — read authoritative values from your monitoring stack.
class MonitoringProvider:
    def get_trusted_context(self):
        return {
            "current_time_ms": float(ntp_client.time_ms()),
            "current_gpu_util": metrics.gpu_utilization(),
            "current_api_util": metrics.api_utilization(),
        }
    @property
    def trusted_field_names(self):
        return frozenset({"current_time_ms", "current_gpu_util", "current_api_util"})

local = NumerailSystemLocal(config, trusted_context_provider=MonitoringProvider())
result = local.enforce(ai_proposal)
```

The provider's values overwrite matching schema fields before the enforcement call.
Fields returned by the provider that are not in the active schema are silently skipped.
A `WARNING` log is emitted for every field where the AI's proposed value differed
from the authoritative value, providing a real-time audit trail of attempted spoofing.

#### Why `current_time_ms` is the most critical trusted field

Every temporal constraint — shift windows, budget period boundaries, freshness
expiry checks, authorization expiry — depends on the wall-clock time.  If the
model can supply its own time value it can simultaneously manipulate all
time-based constraints by asserting an earlier timestamp.  `current_time_ms` is
therefore the default trusted field in `DefaultTimeProvider`.

**Why milliseconds, not nanoseconds.**  `float64` can represent integers exactly
up to 2^53 ≈ 9.0 × 10^15.  The current Unix time in milliseconds is
~1.7 × 10^12, comfortably within range.  In nanoseconds it is ~1.7 × 10^18,
which exceeds the exact-integer range and causes silent rounding errors in
temporal constraints.  Always store and compare wall-clock time as milliseconds
in float64 fields.

#### Fields that should be trusted in production deployments

| Field | Why it must be server-authoritative |
|-------|-------------------------------------|
| `current_time_ms` | Every temporal constraint depends on it; agent spoofing shifts all time-based gates simultaneously |
| `shift_start_ms` | Defines when the current budget period began; agent control lets it reset the clock |
| `shift_end_ms` | Defines when the budget period ends; agent control lets it extend authority indefinitely |
| `gpu_utilization` | Used in headroom constraints; a spoofed low value bypasses the GPU utilisation ceiling |
| `api_utilization` | Same for API headroom constraints |
| `db_utilization` | Same for database utilisation constraints |
| `queue_depth` | Same for queue-depth headroom constraints |
| `error_rate` | Determines whether the system is in a degraded state; agent control masks failures |
| `memory_utilization` | Same as GPU utilisation for memory constraints |
| `controller_gpu_reserve` | If agent-controlled, can be set to zero, eliminating protected controller capacity |
| `controller_memory_reserve` | Same for memory controller reserves |
| `controller_api_reserve` | Same for API controller reserves |
| `budget_gpu_remaining` | Determines the remaining GPU budget; inflation lets the agent overspend |
| `budget_api_remaining` | Same for API budget |
| `budget_mutations_remaining` | Same for cloud mutation budget |
| `breaker_mode` | If agent-controlled, the agent can keep the breaker in CLOSED mode under genuine stress |
| `state_version` | Monotone counter preventing replay attacks using older state |
| `disturbance_margins` | If agent-controlled, can be set to zero, eliminating safety margins |

#### Implementing a production TrustedContextProvider

```python
class ProductionTrustedProvider:
    """Read authoritative values from infrastructure on every call."""

    def get_trusted_context(self) -> dict[str, float]:
        # Use NTP for wall-clock time — never the local clock unchecked.
        current_time = float(ntp_client.synchronized_time_ms())
        # Read utilisation from your monitoring stack (Prometheus, Datadog, etc.)
        return {
            "current_time_ms": current_time,
            "current_gpu_util": prometheus.query("gpu_utilization"),
            "current_api_util": prometheus.query("api_utilization"),
            "current_db_util": prometheus.query("db_utilization"),
            "budget_gpu_remaining": budget_service.remaining("gpu"),
            "budget_api_remaining": budget_service.remaining("api"),
        }

    @property
    def trusted_field_names(self) -> frozenset[str]:
        return frozenset({
            "current_time_ms", "current_gpu_util", "current_api_util",
            "current_db_util", "budget_gpu_remaining", "budget_api_remaining",
        })
```

#### The `trusted_overrides` audit field

Every `enforce()` call returns a `SystemEnforcementResult` whose `trusted_overrides`
attribute maps field names to `(ai_proposed_value, authoritative_value)` tuples for
every field where the provider's value differed from the AI's claim by more than 1e-9.

A non-empty `trusted_overrides` is an observable signal of attempted policy
manipulation: the model proposed a value for a field it is not permitted to control,
and the server's authoritative value was substituted before enforcement.  Log and
alert on non-empty `trusted_overrides` in production; a model that consistently
claims low utilisation when the real value is high is exhibiting a systematic
pressure against its own governance boundary.

```python
result = local.enforce(ai_proposal)
if result.trusted_overrides:
    for field, (ai_val, auth_val) in result.trusted_overrides.items():
        alert(f"Trusted field spoofing: {field}: AI={ai_val:.4f} auth={auth_val:.4f}")
```

## Budget lifecycle

Budgets make the feasible region time-varying. Under nonnegative consumption mode, remaining budget is non-increasing, the targeted bound is non-increasing, and the corresponding half-space is non-expanding. The region shrinks monotonically as resources are consumed.

In the deployed system, budget state must be durable. The engine's in-memory `BudgetTracker` is sufficient for single-process local enforcement, but a production deployment requires:

- Persistent storage of per-budget consumed totals.
- Row-level locking (`SELECT ... FOR UPDATE`) on budget rows to prevent concurrent overspend.
- Persistent storage of per-action budget deltas for rollback.
- Transactional commit of budget consumption alongside the audit record and action ledger entry.

Rollback restores budget state when downstream execution fails. The two-phase protocol is: (1) enforce and consume budget atomically, (2) attempt downstream execution, (3) if execution fails, rollback restores the budget. The `action_id` serves as the idempotency key across the enforcement and execution systems. Rollback is idempotent — a second rollback of the same action_id returns False.

## Outbox pattern

Approved or projected actions with a configured execution topic enqueue a delivery event in the same transaction as the enforcement decision. A separate worker process delivers events to downstream systems. Failed deliveries are retried, then dead-lettered. Rejected actions never enqueue outbox events.

The outbox ensures that enforcement and downstream execution are connected by an at-least-once delivery guarantee, without requiring the downstream system to be available at enforcement time.

## Audit and evidence

Every enforcement decision produces an audit record appended to a SHA-256 hash-linked chain. Each record contains the previous record's hash, making the chain tamper-evident: modifying, inserting, or deleting any record is detectable by chain verification.

Each record contains: result (approve/project/reject), original vector, enforced vector, violated constraints, binding constraints, correction distance, solver method, region version (including budget state digest), timestamp, and action_id.

In a production deployment, the audit chain should be backed by durable storage. The engine's in-memory `AuditChain` provides the hash-linking logic; the deployment wraps it with persistence.

A separate action ledger should store structured records including: action_id, raw proposed action (agent's claim), merged action (after trusted context), enforced action, budget deltas, rollback status, active policy version, and actor/scope metadata. The audit chain provides tamper evidence; the ledger provides queryable operational records.

## Operational metrics

The enforcement stream produces metrics per policy version: approve/project/reject rates, correction distance statistics (mean, p99, max), solver distribution, top violated constraints, and top binding constraints. These constitute an observatory on the AI's decision-making that does not exist without the enforcement layer.

If max_amount violations increase week-over-week, the AI's magnitude calibration is drifting. If max_risk rejections spike, something in the input distribution changed. If the approve rate drops below a threshold, the constraints may need revision. The gap between proposed and enforced actions is diagnostic information with direct operational value.

## End-to-end assurance conditions

The paper's addendum names six conditions under which the engine-level guarantee composes into an end-to-end deployment assurance:

**C1. Checker correctness.** Each `is_satisfied` implementation correctly characterizes membership. (Verified by `verify_proof.py`; empirically tested.)

**C2. Transactional integrity.** The database provides ACID transactions with row-level locking. (Architectural assumption; tested locally.)

**C3. Trusted context integrity.** Server-supplied values are accurate, current, and provided by an authorized orchestrator. (Operational assumption; auditable via the action ledger.)

**C4. Policy version correctness.** The active policy version reflects the intended constraint geometry. (Operational assumption; auditable via version history.)

**C5. API key isolation.** The agent's API key carries only the enforce scope. (Architectural assumption; enforced by the security module.)

**C6. Implementation fidelity.** The Python enforcement path correctly implements the formally specified logic. (Not formally proved; empirically tested by 49 certification tests.)

Under C1–C6, the end-to-end assurance holds: approved and projected actions satisfy every active constraint; budgets are consumed atomically and correctly; rollback restores state; the audit chain is tamper-evident; and the monotone ratchet prevents capability re-escalation by the agent.

If any condition is violated, the assurance degrades in a specific, traceable way. C1 violation: the post-check may approve infeasible points. C2 violation: concurrent requests may overspend budgets. C3 violation: enforcement is correct for the wrong merged action. C4 violation: correct enforcement of incorrect policy. C5 violation: the agent may loosen its own constraints. C6 violation: all properties at risk.

---

## Control-plane reserve pattern

Numerail guarantees membership in F_t: if the decision is APPROVE or PROJECT, the enforced output satisfies every active constraint. If you want the governance controller itself to remain alive under load — so the circuit breaker does not starve its own control loop — define F_t to subtract controller reserve and disturbance margin from admissible workload.

This is a recommended deployment pattern, not an automatic universal property of the engine.

The construction: add trusted fields for the controller's GPU reserve, API reserve, parallelism reserve, and disturbance margins. Replace simple headroom constraints with reserve-aware versions that include the reserve and margin terms. Mark all reserve and margin fields as trusted (so the agent cannot lower them) and forbidden (so the solver cannot alter them). The result: any point in the feasible region, by construction, leaves the declared controller capacity intact.

For example, the simple headroom constraint:

```
current_gpu_util + gpu_seconds / 240 ≤ 0.90
```

becomes the reserve-aware version:

```
current_gpu_util + gpu_seconds / 240 + ctrl_gpu_reserve / 240 + disturbance_margin / 240 ≤ 0.90
```

If the orchestrator injects `ctrl_gpu_reserve = 30` and `disturbance_margin = 15`, the maximum admissible `gpu_seconds` shrinks — because the controller's protected capacity is subtracted before the agent's request is evaluated. A workload that would have passed simple headroom (e.g., `gpu_seconds = 120` yields 0.80 ≤ 0.90) now fails reserve-aware headroom (0.9875 > 0.90).

The stronger property — control-plane survivability — comes from the policy design, not from a different engine theorem. See `examples/ai_circuit_breaker.py` for the complete implementation.

## Defense-in-depth invariant

The `enforce()` function contains a final invariant check inside its `_out()` helper: if the result is APPROVE or PROJECT, the enforced vector is re-verified against the combined checker for the active feasible region. If this check fails, the function raises `AssertionError`.

This exception type is deliberate. The invariant represents an internal consistency violation — something that should never fire in correctly functioning code. Using a domain exception like `NumerailError` would allow it to be silently caught by application-level `except NumerailError` handlers, masking an implementation bug. `AssertionError` is the correct signal: it says "the engine's own logic has a defect" and should propagate to the operator as a crash, not be handled as a normal operational error.

The check uses an explicit `raise` statement (not the `assert` keyword) so it cannot be stripped by `python -O`.

---

