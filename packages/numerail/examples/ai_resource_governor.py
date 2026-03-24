"""Numerail flagship example: AI resource governor.

Demonstrates how Numerail governs the execution envelope of an AI system.
The model may request compute and tool authority, but the runtime only
receives a grant that satisfies declared policy.

The guarantee applies to conformance with the encoded feasible region,
not to the correctness or completeness of the policy specification itself.

Constraint layers:
  - Linear:     per-step ceilings, floors, structural relations, headroom caps
  - Quadratic:  combined resource energy bound (no peak-everything-at-once)
  - SOCP:       burst envelope under live infrastructure conditions
  - PSD / LMI:  coupled headroom — tight GPU/API leaves no room for fan-out

Trusted fields:
  - current_gpu_util and current_api_util are server-measured telemetry,
    not agent-controlled. The agent cannot lower its own admissibility barrier.

Budgets:
  - gpu_day:  cumulative GPU seconds over the day (monotone non-expanding)
  - api_day:  cumulative external API calls over the day

This example uses NumerailSystemLocal, which exercises the full production
code path: policy parse, scope checks, trusted-context injection, budget
handling, ledger rows, audit hash chain, outbox enqueue, and rollback.
"""

import numpy as np
from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════
#  POLICY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

fields = [
    "prompt_k",           # prompt-token ceiling, thousands
    "completion_k",       # completion-token ceiling, thousands
    "tool_calls",         # total tool-call allowance for this step
    "external_api_calls", # paid / side-effecting API calls
    "gpu_seconds",        # GPU lease for this step
    "parallel_workers",   # max concurrent subtasks
    "current_gpu_util",   # server-trusted instantaneous GPU utilization [0,1]
    "current_api_util",   # server-trusted instantaneous API utilization [0,1]
]

n = len(fields)
idx = {f: i for i, f in enumerate(fields)}


# ── Linear constraints (Ax ≤ b) ─────────────────────────────────────

def row(coeffs):
    """Build a dense row from {field: coeff} dict."""
    r = [0.0] * n
    for f, v in coeffs.items():
        r[idx[f]] = v
    return r


maxes = {
    "prompt_k": 64, "completion_k": 16, "tool_calls": 40,
    "external_api_calls": 20, "gpu_seconds": 120, "parallel_workers": 16,
    "current_gpu_util": 1, "current_api_util": 1,
}

A_rows, b_rows, names = [], [], []

# Upper bounds
for f in fields:
    A_rows.append(row({f: 1.0}))
    b_rows.append(float(maxes[f]))
    names.append(f"max_{f}")

# Lower bounds (all ≥ 0)
for f in fields:
    A_rows.append(row({f: -1.0}))
    b_rows.append(0.0)
    names.append(f"min_{f}")

# Structural: external_api_calls ≤ tool_calls
A_rows.append(row({"external_api_calls": 1, "tool_calls": -1}))
b_rows.append(0.0)
names.append("external_le_tool_calls")

# Headroom: current_gpu_util + gpu_seconds/240 ≤ 0.90
A_rows.append(row({"current_gpu_util": 1, "gpu_seconds": 1 / 240}))
b_rows.append(0.90)
names.append("gpu_headroom_linear")

# Headroom: current_api_util + external_api_calls/40 ≤ 0.90
A_rows.append(row({"current_api_util": 1, "external_api_calls": 1 / 40}))
b_rows.append(0.90)
names.append("api_headroom_linear")

# Daily budget target rows (separate from per-step maxes)
A_rows.append(row({"gpu_seconds": 1}))
b_rows.append(600.0)
names.append("remaining_gpu_day")

A_rows.append(row({"external_api_calls": 1}))
b_rows.append(80.0)
names.append("remaining_api_day")


# ── Quadratic constraint: resource energy ────────────────────────────
# Prevents the agent from maxing out every expensive dimension at once.
# (prompt_k/64)² + (completion_k/16)² + (tool_calls/40)² +
# (external_api/20)² + (gpu_sec/120)² + (workers/16)² ≤ 2.25

Q_diag = [1 / 64**2, 1 / 16**2, 1 / 40**2, 1 / 20**2,
          1 / 120**2, 1 / 16**2, 0, 0]


# ── SOCP constraint: burst envelope ──────────────────────────────────
# ‖[gpu_util + gpu_sec/240, api_util + ext_api/40, workers/16]‖₂ ≤ 1.15
# Under high live utilization, even moderate requests become infeasible.

M_socp = np.zeros((3, n))
M_socp[0, idx["current_gpu_util"]] = 1.0
M_socp[0, idx["gpu_seconds"]] = 1 / 240
M_socp[1, idx["current_api_util"]] = 1.0
M_socp[1, idx["external_api_calls"]] = 1 / 40
M_socp[2, idx["parallel_workers"]] = 1 / 16


# ── PSD constraint: coupled headroom ─────────────────────────────────
# A(x) = [[1 - (gpu_util + gpu_sec/240),  0.35·workers/16],
#          [0.35·workers/16,               1 - (api_util + ext_api/40)]]  ≽ 0
# When GPU/API headroom is tight, aggressive fan-out is blocked.

A0_psd = np.eye(2).tolist()
A_psd_list = [
    [[0, 0], [0, 0]],                             # prompt_k
    [[0, 0], [0, 0]],                             # completion_k
    [[0, 0], [0, 0]],                             # tool_calls
    [[0, 0], [0, -1 / 40]],                       # external_api_calls
    [[-1 / 240, 0], [0, 0]],                      # gpu_seconds
    [[0, 0.35 / 16], [0.35 / 16, 0]],             # parallel_workers
    [[-1, 0], [0, 0]],                            # current_gpu_util
    [[0, 0], [0, -1]],                            # current_api_util
]


# ── Assemble config ──────────────────────────────────────────────────

config = {
    "policy_id": "ai_resource_governor_v1",

    "schema": {"fields": fields},

    "polytope": {
        "A": [list(r) for r in A_rows],
        "b": b_rows,
        "names": names,
    },

    "quadratic_constraints": [{
        "Q": np.diag(Q_diag).tolist(),
        "a": [0] * n,
        "b": 2.25,
        "name": "resource_energy",
    }],

    "socp_constraints": [{
        "M": M_socp.tolist(),
        "q": [0] * 3,
        "c": [0] * n,
        "d": 1.15,
        "name": "burst_envelope",
    }],

    "psd_constraints": [{
        "A0": A0_psd,
        "A_list": A_psd_list,
        "name": "coupled_headroom_psd",
    }],

    "trusted_fields": ["current_gpu_util", "current_api_util"],

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
    },

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
    ],
}


# ═══════════════════════════════════════════════════════════════════════
#  CONSTRUCT RUNTIME
# ═══════════════════════════════════════════════════════════════════════

local = NumerailSystemLocal(config)


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

scenarios = [
    {
        "name": "Safe baseline grant",
        "proposed": {
            "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
            "external_api_calls": 5, "gpu_seconds": 60, "parallel_workers": 4,
            "current_gpu_util": 0.30, "current_api_util": 0.20,
        },
        "trusted": {"current_gpu_util": 0.30, "current_api_util": 0.20},
        "topic": "runtime",
    },
    {
        "name": "Telemetry spoof attempt",
        "proposed": {
            "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
            "external_api_calls": 5, "gpu_seconds": 80, "parallel_workers": 4,
            "current_gpu_util": 0.10, "current_api_util": 0.10,
        },
        "trusted": {"current_gpu_util": 0.65, "current_api_util": 0.55},
        "topic": "runtime",
    },
    {
        "name": "Overlarge GPU burst (system hot)",
        "proposed": {
            "prompt_k": 48, "completion_k": 12, "tool_calls": 20,
            "external_api_calls": 10, "gpu_seconds": 120, "parallel_workers": 8,
            "current_gpu_util": 0.50, "current_api_util": 0.50,
        },
        "trusted": {"current_gpu_util": 0.80, "current_api_util": 0.75},
        "topic": "runtime",
    },
    {
        "name": "External API calls exceed tool calls",
        "proposed": {
            "prompt_k": 16, "completion_k": 4, "tool_calls": 4,
            "external_api_calls": 8, "gpu_seconds": 30, "parallel_workers": 2,
            "current_gpu_util": 0.20, "current_api_util": 0.20,
        },
        "trusted": {"current_gpu_util": 0.20, "current_api_util": 0.20},
        "topic": "runtime",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  ENFORCEMENT LOOP
# ═══════════════════════════════════════════════════════════════════════

print("=" * 74)
print("  Numerail AI Resource Governor")
print("=" * 74)
print()

for i, sc in enumerate(scenarios):
    action_id = f"grant_{i + 1}"
    name = sc["name"]
    proposed = sc["proposed"]
    trusted = sc["trusted"]
    topic = sc["topic"]

    result = local.enforce(
        proposed,
        action_id=action_id,
        trusted_context=trusted,
        execution_topic=topic,
    )

    decision = result["decision"]
    feedback = result["feedback"]
    enforced = result["enforced_values"]

    print(f"  Scenario {i + 1}: {name}")
    print(f"  Action ID: {action_id}")
    print()

    # Show proposed vs trusted telemetry
    print(f"    Proposed grant:")
    for f in fields[:6]:
        print(f"      {f:24s} = {proposed[f]}")
    print(f"      {'current_gpu_util':24s} = {proposed['current_gpu_util']}"
          f"  (agent claims)")
    print(f"      {'current_api_util':24s} = {proposed['current_api_util']}"
          f"  (agent claims)")
    print()

    merged = feedback.get("merged_values")
    if merged and (merged.get("current_gpu_util") != proposed["current_gpu_util"]
                   or merged.get("current_api_util") != proposed["current_api_util"]):
        print(f"    Trusted telemetry override:")
        print(f"      current_gpu_util: {proposed['current_gpu_util']}"
              f" -> {merged['current_gpu_util']}  (server-measured)")
        print(f"      current_api_util: {proposed['current_api_util']}"
              f" -> {merged['current_api_util']}  (server-measured)")
        print()

    print(f"    Decision: {decision.upper()}")

    if decision in ("approve", "project") and enforced:
        print(f"    Enforced grant:")
        for f in fields:
            val = enforced[f]
            orig = proposed[f] if not merged else merged.get(f, proposed[f])
            marker = ""
            if abs(val - orig) > 0.01:
                marker = f"  (was {orig:.1f})"
            print(f"      {f:24s} = {val:8.2f}{marker}")

    violations = feedback.get("violations", [])
    if violations:
        names_str = ", ".join(v["constraint"] for v in violations)
        print(f"    Violated: {names_str}")

    dist = feedback.get("distance", 0)
    if dist > 0:
        print(f"    Distance: {dist:.4f}")

    routing = feedback.get("routing")
    if routing:
        print(f"    Routing:  {routing}")

    flagged = feedback.get("flagged_dimensions")
    if flagged:
        print(f"    Flagged dims: {flagged}")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  BUDGET STATUS
# ═══════════════════════════════════════════════════════════════════════

print("-" * 74)
budgets = local.budget_remaining
print(f"  GPU day budget:  {600.0 - budgets.get('gpu_day', 600.0):.1f}"
      f" / 600.0 seconds consumed"
      f"   (remaining: {budgets.get('gpu_day', 0):.1f})")
print(f"  API day budget:  {80.0 - budgets.get('api_day', 80.0):.1f}"
      f" / 80.0 calls consumed"
      f"     (remaining: {budgets.get('api_day', 0):.1f})")
print()


# ═══════════════════════════════════════════════════════════════════════
#  OUTBOX STATUS
# ═══════════════════════════════════════════════════════════════════════

print(f"  Outbox events: {len(local.outbox_events)}")
for evt in local.outbox_events:
    print(f"    action_id={evt['action_id']}, topic={evt['topic']}")
print()


# ═══════════════════════════════════════════════════════════════════════
#  ROLLBACK DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════

print("  Rolling back grant_1 (safe baseline)...")
rb = local.rollback("grant_1")
print(f"  Rollback result: rolled_back={rb.rolled_back}, audit_hash={rb.audit_hash[:16]}...")
budgets_after = local.budget_remaining
print(f"  GPU day after rollback: remaining={budgets_after.get('gpu_day', 0):.1f}"
      f"  (restored {600.0 - budgets_after.get('gpu_day', 0):.1f} consumed)")
print(f"  API day after rollback: remaining={budgets_after.get('api_day', 0):.1f}"
      f"  (restored {80.0 - budgets_after.get('api_day', 0):.1f} consumed)")
print()


# ═══════════════════════════════════════════════════════════════════════
#  AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════

print(f"  Audit records: {len(local.audit_records)}")
for rec in local.audit_records:
    print(f"    type={rec['type']}, hash={rec['hash'][:16]}...")
print()

print("=" * 74)
print("  THE GUARANTEE: if the decision is APPROVE or PROJECT, the enforced")
print("  grant satisfies every active constraint. The agent cannot obtain")
print("  runtime authority outside the declared policy.")
print("=" * 74)
