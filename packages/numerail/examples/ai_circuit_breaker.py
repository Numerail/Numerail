"""Numerail advanced example: AI circuit breaker with control-plane reserve.

Demonstrates a deployment pattern where the feasible region explicitly
subtracts trusted controller reserve and disturbance margin from admissible
workload authority. The result: Numerail's existing guarantee automatically
implies that any approved or projected action preserves enough capacity
for the governance system itself to keep running.

This is a POLICY PATTERN, not an engine feature. The engine theorem is
unchanged: APPROVE or PROJECT implies y ∈ F_t. The stronger property —
control-plane survivability — holds because F_t is constructed to encode it.

Key extension over the base AI resource governor:
  - 5 new trusted fields: controller GPU/API/parallelism reserves,
    plus GPU/API disturbance margins for bounded uncertainty
  - Reserve-aware headroom constraints that subtract controller capacity
    from the admissible workload envelope
  - Reserve-aware SOCP and PSD constraints that judge burst footprint
    and fan-out coupling after protecting the controller

Run after examples/ai_resource_governor.py to see the progression from
base governance to control-survivable governance.
"""

import numpy as np
from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════
#  SCHEMA: workload + telemetry + control-plane reserves
# ═══════════════════════════════════════════════════════════════════════

fields = [
    # Workload request (agent-proposed)
    "prompt_k",                        # prompt-token ceiling, thousands
    "completion_k",                    # completion-token ceiling, thousands
    "tool_calls",                      # total tool-call allowance
    "external_api_calls",              # paid / side-effecting API calls
    "gpu_seconds",                     # GPU lease for this step
    "parallel_workers",                # max concurrent subtasks
    # Trusted telemetry (server-measured)
    "current_gpu_util",                # instantaneous GPU utilization [0,1]
    "current_api_util",                # instantaneous API utilization [0,1]
    # Control-plane reserves (trusted policy terms)
    "ctrl_gpu_reserve_seconds",        # GPU capacity reserved for governance
    "ctrl_api_reserve_calls",          # API capacity reserved for governance
    "ctrl_parallel_reserve",           # parallelism slots reserved for governance
    # Disturbance margins (trusted robustness terms)
    "gpu_disturbance_margin_seconds",  # GPU robustness margin
    "api_disturbance_margin_calls",    # API robustness margin
]

n = len(fields)
idx = {f: i for i, f in enumerate(fields)}


# ═══════════════════════════════════════════════════════════════════════
#  LINEAR CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

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
    "ctrl_gpu_reserve_seconds": 40, "ctrl_api_reserve_calls": 6,
    "ctrl_parallel_reserve": 2,
    "gpu_disturbance_margin_seconds": 20, "api_disturbance_margin_calls": 4,
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

# ── Reserve-aware headroom constraints ───────────────────────────────
# These are the core of the pattern. They convert "do not overload the
# system" into "do not overload the system after reserving capacity for
# the governance controller and disturbance margin."

# GPU: util + workload/240 + reserve/240 + margin/240 ≤ 0.90
A_rows.append(row({
    "current_gpu_util": 1,
    "gpu_seconds": 1 / 240,
    "ctrl_gpu_reserve_seconds": 1 / 240,
    "gpu_disturbance_margin_seconds": 1 / 240,
}))
b_rows.append(0.90)
names.append("gpu_headroom_reserve")

# API: util + workload/40 + reserve/40 + margin/40 ≤ 0.90
A_rows.append(row({
    "current_api_util": 1,
    "external_api_calls": 1 / 40,
    "ctrl_api_reserve_calls": 1 / 40,
    "api_disturbance_margin_calls": 1 / 40,
}))
b_rows.append(0.90)
names.append("api_headroom_reserve")

# Parallelism: workers + ctrl_reserve ≤ 16
A_rows.append(row({
    "parallel_workers": 1,
    "ctrl_parallel_reserve": 1,
}))
b_rows.append(16.0)
names.append("parallel_headroom_reserve")

# Daily budget target rows
A_rows.append(row({"gpu_seconds": 1}))
b_rows.append(600.0)
names.append("remaining_gpu_day")

A_rows.append(row({"external_api_calls": 1}))
b_rows.append(80.0)
names.append("remaining_api_day")


# ═══════════════════════════════════════════════════════════════════════
#  QUADRATIC CONSTRAINT: workload energy (unchanged from base)
# ═══════════════════════════════════════════════════════════════════════

Q_diag = [1 / 64**2, 1 / 16**2, 1 / 40**2, 1 / 20**2,
          1 / 120**2, 1 / 16**2] + [0] * 7  # zeros on all trusted dims


# ═══════════════════════════════════════════════════════════════════════
#  SOCP CONSTRAINT: reserve-aware burst envelope
# ═══════════════════════════════════════════════════════════════════════
# Total load including reserves must stay inside the cone.

M_socp = np.zeros((3, n))
M_socp[0, idx["current_gpu_util"]] = 1.0
M_socp[0, idx["gpu_seconds"]] = 1 / 240
M_socp[0, idx["ctrl_gpu_reserve_seconds"]] = 1 / 240
M_socp[0, idx["gpu_disturbance_margin_seconds"]] = 1 / 240
M_socp[1, idx["current_api_util"]] = 1.0
M_socp[1, idx["external_api_calls"]] = 1 / 40
M_socp[1, idx["ctrl_api_reserve_calls"]] = 1 / 40
M_socp[1, idx["api_disturbance_margin_calls"]] = 1 / 40
M_socp[2, idx["parallel_workers"]] = 1 / 16
M_socp[2, idx["ctrl_parallel_reserve"]] = 1 / 16


# ═══════════════════════════════════════════════════════════════════════
#  PSD CONSTRAINT: reserve-aware coupled headroom
# ═══════════════════════════════════════════════════════════════════════
# Fan-out coupling is judged after subtracting controller reserve.

A0_psd = np.eye(2).tolist()
A_psd_list = [[[0, 0], [0, 0]] for _ in range(n)]

# GPU headroom diagonal: workload + reserve + margin
A_psd_list[idx["gpu_seconds"]] = [[-1 / 240, 0], [0, 0]]
A_psd_list[idx["ctrl_gpu_reserve_seconds"]] = [[-1 / 240, 0], [0, 0]]
A_psd_list[idx["gpu_disturbance_margin_seconds"]] = [[-1 / 240, 0], [0, 0]]
A_psd_list[idx["current_gpu_util"]] = [[-1, 0], [0, 0]]

# API headroom diagonal: workload + reserve + margin
A_psd_list[idx["external_api_calls"]] = [[0, 0], [0, -1 / 40]]
A_psd_list[idx["ctrl_api_reserve_calls"]] = [[0, 0], [0, -1 / 40]]
A_psd_list[idx["api_disturbance_margin_calls"]] = [[0, 0], [0, -1 / 40]]
A_psd_list[idx["current_api_util"]] = [[0, 0], [0, -1]]

# Concurrency coupling off-diagonal: workload + reserve
A_psd_list[idx["parallel_workers"]] = [[0, 0.35 / 16], [0.35 / 16, 0]]
A_psd_list[idx["ctrl_parallel_reserve"]] = [[0, 0.35 / 16], [0.35 / 16, 0]]


# ═══════════════════════════════════════════════════════════════════════
#  ASSEMBLE CONFIG
# ═══════════════════════════════════════════════════════════════════════

config = {
    "policy_id": "ai_circuit_breaker_v1",

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

    "trusted_fields": [
        "current_gpu_util", "current_api_util",
        "ctrl_gpu_reserve_seconds", "ctrl_api_reserve_calls",
        "ctrl_parallel_reserve",
        "gpu_disturbance_margin_seconds", "api_disturbance_margin_calls",
    ],

    "enforcement": {
        "mode": "project",
        "dimension_policies": {
            "current_gpu_util": "forbidden",
            "current_api_util": "forbidden",
            "ctrl_gpu_reserve_seconds": "forbidden",
            "ctrl_api_reserve_calls": "forbidden",
            "ctrl_parallel_reserve": "forbidden",
            "gpu_disturbance_margin_seconds": "forbidden",
            "api_disturbance_margin_calls": "forbidden",
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
#  TRUSTED CONTEXT (server-injected)
# ═══════════════════════════════════════════════════════════════════════

# These values are what the orchestrator injects. The agent cannot
# lower any of them to make its own request admissible.
default_trusted = {
    "current_gpu_util": 0.30,
    "current_api_util": 0.20,
    "ctrl_gpu_reserve_seconds": 30,
    "ctrl_api_reserve_calls": 4,
    "ctrl_parallel_reserve": 2,
    "gpu_disturbance_margin_seconds": 15,
    "api_disturbance_margin_calls": 3,
}

# Lighter reserves for scenarios that need projection room
light_trusted = {
    "current_gpu_util": 0.20,
    "current_api_util": 0.20,
    "ctrl_gpu_reserve_seconds": 15,
    "ctrl_api_reserve_calls": 2,
    "ctrl_parallel_reserve": 1,
    "gpu_disturbance_margin_seconds": 8,
    "api_disturbance_margin_calls": 2,
}


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: compute and print reserve status
# ═══════════════════════════════════════════════════════════════════════

workload_fields = fields[:6]


def print_reserve_status(enforced, trusted):
    """Print explicit remaining headroom after the admitted action."""
    gpu_used = (trusted["current_gpu_util"]
                + enforced["gpu_seconds"] / 240
                + trusted["ctrl_gpu_reserve_seconds"] / 240
                + trusted["gpu_disturbance_margin_seconds"] / 240)
    api_used = (trusted["current_api_util"]
                + enforced["external_api_calls"] / 40
                + trusted["ctrl_api_reserve_calls"] / 40
                + trusted["api_disturbance_margin_calls"] / 40)
    par_used = enforced["parallel_workers"] + trusted["ctrl_parallel_reserve"]

    gpu_ok = gpu_used <= 0.90 + 1e-6
    api_ok = api_used <= 0.90 + 1e-6
    par_ok = par_used <= 16.0 + 1e-6

    print(f"    Reserve status:")
    print(f"      GPU headroom:  {gpu_used:.4f} / 0.90  "
          f"({'preserved' if gpu_ok else 'VIOLATED'})")
    print(f"      API headroom:  {api_used:.4f} / 0.90  "
          f"({'preserved' if api_ok else 'VIOLATED'})")
    print(f"      Parallel:      {par_used:.0f} / 16     "
          f"({'preserved' if par_ok else 'VIOLATED'})")


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

scenarios = [
    {
        "name": "Safe grant with reserve preserved",
        "proposed": {
            "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
            "external_api_calls": 5, "gpu_seconds": 60, "parallel_workers": 4,
        },
        "trusted": default_trusted,
        "topic": "runtime",
    },
    {
        "name": "Reserve spoof attempt",
        "proposed": {
            "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
            "external_api_calls": 5, "gpu_seconds": 100, "parallel_workers": 4,
        },
        "agent_fake_trusted": {
            "current_gpu_util": 0.10, "current_api_util": 0.10,
            "ctrl_gpu_reserve_seconds": 5, "ctrl_api_reserve_calls": 1,
            "ctrl_parallel_reserve": 0,
            "gpu_disturbance_margin_seconds": 2, "api_disturbance_margin_calls": 1,
        },
        "trusted": default_trusted,
        "topic": "runtime",
    },
    {
        "name": "Would pass old policy, fails reserve-aware policy",
        "proposed": {
            "prompt_k": 32, "completion_k": 8, "tool_calls": 10,
            "external_api_calls": 5, "gpu_seconds": 120, "parallel_workers": 4,
        },
        "trusted": default_trusted,
        "topic": "runtime",
    },
    {
        "name": "Structural projection with reserve intact",
        "proposed": {
            "prompt_k": 16, "completion_k": 4, "tool_calls": 4,
            "external_api_calls": 8, "gpu_seconds": 30, "parallel_workers": 2,
        },
        "trusted": light_trusted,
        "topic": "runtime",
    },
    {
        "name": "Boundary case (GPU headroom binds at limit)",
        "proposed": {
            "prompt_k": 16, "completion_k": 4, "tool_calls": 5,
            "external_api_calls": 2, "gpu_seconds": 99.0, "parallel_workers": 4,
        },
        "trusted": default_trusted,
        "topic": "runtime",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  ENFORCEMENT LOOP
# ═══════════════════════════════════════════════════════════════════════

print("=" * 74)
print("  Numerail AI Circuit Breaker — Control-Plane Reserve Pattern")
print("=" * 74)
print()

for i, sc in enumerate(scenarios):
    action_id = f"cb_{i + 1}"
    name = sc["name"]
    proposed = sc["proposed"]
    trusted = sc["trusted"]
    topic = sc["topic"]

    # Build the full action dict (agent's workload + agent's claimed trusted fields)
    if "agent_fake_trusted" in sc:
        action = {**proposed, **sc["agent_fake_trusted"]}
    else:
        action = {**proposed, **trusted}

    result = local.enforce(
        action,
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

    # Show proposed workload
    print(f"    Proposed workload:")
    for f in workload_fields:
        print(f"      {f:24s} = {proposed[f]}")
    print()

    # Show trusted context
    print(f"    Trusted context (server-injected):")
    for f in sorted(trusted.keys()):
        print(f"      {f:38s} = {trusted[f]}")

    # Show spoof detection if applicable
    if "agent_fake_trusted" in sc:
        fake = sc["agent_fake_trusted"]
        print()
        print(f"    Spoof detection:")
        for f in sorted(fake.keys()):
            if fake[f] != trusted[f]:
                print(f"      {f}: agent claimed {fake[f]}, "
                      f"server injected {trusted[f]}")
    print()

    print(f"    Decision: {decision.upper()}")

    if decision in ("approve", "project") and enforced:
        print(f"    Enforced workload:")
        for f in workload_fields:
            val = enforced[f]
            orig = proposed[f]
            marker = ""
            if abs(val - orig) > 0.01:
                marker = f"  (was {orig:.1f})"
            print(f"      {f:24s} = {val:8.2f}{marker}")
        print()
        print_reserve_status(enforced, trusted)

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

    # Scenario 3: explain the reserve-aware rejection
    if i == 2 and decision == "reject":
        gpu_old = 0.30 + 120 / 240
        gpu_new = 0.30 + 120 / 240 + 30 / 240 + 15 / 240
        print()
        print(f"    Without reserve:  gpu_headroom = {gpu_old:.4f} <= 0.90 (PASS)")
        print(f"    With reserve:     gpu_headroom = {gpu_new:.4f} > 0.90  (FAIL)")
        print(f"    The reserve-aware policy blocks workloads that would")
        print(f"    starve the governance controller's protected capacity.")

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
#  ROLLBACK
# ═══════════════════════════════════════════════════════════════════════

print("  Rolling back cb_1 (safe baseline)...")
rb = local.rollback("cb_1")
print(f"  Rollback: rolled_back={rb.rolled_back}, audit_hash={rb.audit_hash[:16]}...")
budgets_after = local.budget_remaining
print(f"  GPU day after rollback: remaining={budgets_after.get('gpu_day', 0):.1f}")
print(f"  API day after rollback: remaining={budgets_after.get('api_day', 0):.1f}")
print()

print("=" * 74)
print("  THE GUARANTEE: if the decision is APPROVE or PROJECT, the enforced")
print("  grant satisfies every constraint — including the reserve-aware")
print("  headroom constraints that protect the governance controller.")
print()
print("  Control survivability comes from the policy design, not from a")
print("  different engine theorem. The engine guarantees y ∈ F_t. The")
print("  deployer constructs F_t to encode the controller's protected reserve.")
print("=" * 74)
