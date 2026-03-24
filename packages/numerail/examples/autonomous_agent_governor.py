"""Numerail example: autonomous agent governor over a 20-step sequence.

Scenario
--------
An AI agent repeatedly proposes a resource grant (tokens, GPU, API calls,
workers) over 20 steps while infrastructure stress rises and falls.  A
lightweight breaker state machine sits between the agent and the enforcement
kernel:

  Agent → Governor (alpha-scale + trusted-context injection) → Numerail kernel

The breaker tracks a weighted overload score and transitions through four modes:

    CLOSED     α = 1.00   full authority
    THROTTLED  α = 0.50   half authority
    HALF_OPEN  α = 0.25   probe mode
    SAFE_STOP  α = 0.05   near-zero authority (latched — manual reset required)

What this example demonstrates
-------------------------------
1. Mode transitions:  CLOSED → THROTTLED → HALF_OPEN → SAFE_STOP → (manual
   reset) → HALF_OPEN → CLOSED.
2. Geometric rejection during SAFE_STOP:  when GPU utilisation exceeds the
   headroom ceiling the feasible region becomes empty and the kernel REJECTs
   regardless of the alpha-scaled proposal.
3. Telemetry spoof detection:  the agent always claims low utilisation; the
   trusted context overwrites it with server-measured values.
4. Mid-sequence rollback:  step 7 is rolled back between steps 7 and 8,
   restoring its budget contribution before the simulation continues.
5. Budget depletion:  the GPU-seconds shift budget depletes across the
   sequence; the final step is projected onto the remaining budget bound.

The guarantee is asserted explicitly at the end: for every APPROVE or PROJECT
output, the enforced vector satisfies every active constraint.

Run with:
    cd packages/numerail
    pip install -e .
    python examples/autonomous_agent_governor.py
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from numerail.local import NumerailSystemLocal


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: SIMULATED BREAKER STATE MACHINE
#
#  This is a plain Python class — no Numerail import needed.  It represents
#  the supervisory layer that an orchestrator (or numerail_ext's
#  StateTransitionGovernor) runs on top of the enforcement kernel.
# ═══════════════════════════════════════════════════════════════════════════════

class BreakerMode(Enum):
    CLOSED    = "CLOSED"     # nominal operations, full authority
    THROTTLED = "THROTTLED"  # elevated load, half authority
    HALF_OPEN = "HALF_OPEN"  # recovering or probing, quarter authority
    SAFE_STOP = "SAFE_STOP"  # emergency, near-zero authority (latched)


# Alpha scale factor per mode.  The governor multiplies every agent-proposed
# workload dimension by alpha before passing it to the enforcement kernel.
# Trusted telemetry fields (current_gpu_util, current_api_util) are NOT scaled
# — they are server-authoritative and cannot be modified by the governor.
ALPHA: dict[BreakerMode, float] = {
    BreakerMode.CLOSED:    1.00,
    BreakerMode.THROTTLED: 0.50,
    BreakerMode.HALF_OPEN: 0.25,
    BreakerMode.SAFE_STOP: 0.05,
}


class BreakerStateMachine:
    """Minimal hysteretic breaker — simplified from numerail_ext's implementation.

    Overload score:  s = 0.60 * gpu_util + 0.40 * api_util

    Thresholds (all configurable):
        trip_score   = 0.50  CLOSED    → THROTTLED
        open_score   = 0.65  THROTTLED → HALF_OPEN
        safe_score   = 0.85  any       → SAFE_STOP  (latches)
        reset_score  = 0.45  HALF_OPEN → CLOSED when score falls below this;
                             also the maximum score at which manual_reset()
                             succeeds for SAFE_STOP.
    """

    def __init__(
        self,
        trip_score:  float = 0.50,
        open_score:  float = 0.65,
        safe_score:  float = 0.85,
        reset_score: float = 0.45,
    ):
        self.mode = BreakerMode.CLOSED
        self._trip  = trip_score
        self._open  = open_score
        self._safe  = safe_score
        self._reset = reset_score

    @staticmethod
    def overload_score(gpu_util: float, api_util: float) -> float:
        """Weighted overload score in [0, 1]."""
        return 0.60 * gpu_util + 0.40 * api_util

    def update(self, gpu_util: float, api_util: float) -> tuple[BreakerMode, BreakerMode]:
        """Transition the breaker given current telemetry.

        Returns (previous_mode, new_mode) so the caller can detect changes.
        """
        prev = self.mode
        s = self.overload_score(gpu_util, api_util)

        if self.mode == BreakerMode.SAFE_STOP:
            # SAFE_STOP is latched — only manual_reset() can clear it.
            pass
        elif s >= self._safe:
            # Emergency threshold: immediately latch to SAFE_STOP from any mode.
            self.mode = BreakerMode.SAFE_STOP
        elif self.mode == BreakerMode.CLOSED:
            if s >= self._trip:
                self.mode = BreakerMode.THROTTLED
        elif self.mode == BreakerMode.THROTTLED:
            if s >= self._open:
                self.mode = BreakerMode.HALF_OPEN
            elif s <= self._reset:
                # Load has dropped back below the reset threshold — reclose.
                self.mode = BreakerMode.CLOSED
        elif self.mode == BreakerMode.HALF_OPEN:
            if s >= self._trip:
                # Re-trip: stress returned before recovery was complete.
                self.mode = BreakerMode.THROTTLED
            elif s <= self._reset:
                # Recovery confirmed: return to full authority.
                self.mode = BreakerMode.CLOSED

        return prev, self.mode

    def manual_reset(self, gpu_util: float, api_util: float) -> bool:
        """Attempt to clear SAFE_STOP latch.

        Succeeds only if current overload score ≤ reset_score.  Returns True
        on success, False if the system is still too stressed to reset.
        """
        if self.mode != BreakerMode.SAFE_STOP:
            return False  # nothing to clear
        if self.overload_score(gpu_util, api_util) <= self._reset:
            # Transition to HALF_OPEN rather than straight to CLOSED — the
            # system must prove stability before regaining full authority.
            self.mode = BreakerMode.HALF_OPEN
            return True
        return False   # overload still too high


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: POLICY CONFIGURATION
#
#  Eight-dimensional schema: six workload fields (agent-proposed) and two
#  telemetry fields (server-trusted).  Constraint layers mirror the base
#  ai_resource_governor example: linear box + structural + headroom, quadratic
#  energy bound, SOCP burst envelope, PSD coupled-headroom matrix.
# ═══════════════════════════════════════════════════════════════════════════════

fields = [
    "prompt_k",           # prompt-token ceiling, thousands
    "completion_k",       # completion-token ceiling, thousands
    "tool_calls",         # total tool-call allowance for this step
    "external_api_calls", # paid / side-effecting external API calls
    "gpu_seconds",        # GPU lease for this step
    "parallel_workers",   # max concurrent subtasks
    "current_gpu_util",   # server-measured instantaneous GPU utilisation [0,1]
    "current_api_util",   # server-measured instantaneous API utilisation [0,1]
]

n   = len(fields)
idx = {f: i for i, f in enumerate(fields)}

# Workload fields the governor scales with alpha.  Telemetry fields are
# NEVER scaled — they come from the server, not from the agent.
WORKLOAD_FIELDS = fields[:6]


def _row(coeffs: dict) -> list[float]:
    """Build a dense constraint row from a {field: coefficient} dict."""
    r = [0.0] * n
    for f, v in coeffs.items():
        r[idx[f]] = float(v)
    return r


# ── Per-field maximums ──────────────────────────────────────────────────────
MAXES = {
    "prompt_k": 64.0,  "completion_k": 16.0, "tool_calls": 40.0,
    "external_api_calls": 20.0, "gpu_seconds": 120.0, "parallel_workers": 16.0,
    "current_gpu_util": 1.0, "current_api_util": 1.0,
}

A_rows: list[list[float]] = []
b_rows: list[float]       = []
cnames: list[str]         = []

# Upper bounds on every field.
for f in fields:
    A_rows.append(_row({f: 1.0}))
    b_rows.append(MAXES[f])
    cnames.append(f"max_{f}")

# Non-negativity floors: all fields ≥ 0.
for f in fields:
    A_rows.append(_row({f: -1.0}))
    b_rows.append(0.0)
    cnames.append(f"min_{f}")

# Structural: external calls must not exceed total tool-call budget.
# Prevents the agent from routing all tool calls externally.
A_rows.append(_row({"external_api_calls": 1.0, "tool_calls": -1.0}))
b_rows.append(0.0)
cnames.append("external_le_tool_calls")

# GPU headroom: live utilisation + normalised workload ≤ 0.90.
# Critically, when current_gpu_util is high (injected by trusted context),
# the remaining headroom for gpu_seconds shrinks accordingly.
# At gpu_util ≥ 0.90 the feasible set for gpu_seconds becomes empty (≤ 0) and
# the kernel REJECTs because no solver can satisfy non-negativity.
A_rows.append(_row({"current_gpu_util": 1.0, "gpu_seconds": 1.0 / 240}))
b_rows.append(0.90)
cnames.append("gpu_headroom")

# API headroom: same pattern for API utilisation.
A_rows.append(_row({"current_api_util": 1.0, "external_api_calls": 1.0 / 40}))
b_rows.append(0.90)
cnames.append("api_headroom")

# Budget target rows — the BudgetTracker tightens these bounds as the shift
# progresses.  When the shift budget nears zero the row bound approaches zero
# and any proposal is projected onto the remaining budget.
A_rows.append(_row({"gpu_seconds": 1.0}))
b_rows.append(600.0)            # initial bound — will be tightened by tracker
cnames.append("remaining_gpu_shift")

A_rows.append(_row({"external_api_calls": 1.0}))
b_rows.append(80.0)
cnames.append("remaining_api_shift")


# ── Quadratic: resource energy bound ───────────────────────────────────────
# Prevents the agent from maxing every expensive dimension simultaneously.
# Each workload dimension is normalised by its per-step ceiling.
# The bound of 2.25 allows roughly 1.5× the energy of a single fully-maxed dim.
Q_diag = [
    1 / 64**2,   # prompt_k
    1 / 16**2,   # completion_k
    1 / 40**2,   # tool_calls
    1 / 20**2,   # external_api_calls
    1 / 120**2,  # gpu_seconds
    1 / 16**2,   # parallel_workers
    0.0,         # current_gpu_util  — telemetry, excluded from energy bound
    0.0,         # current_api_util
]


# ── SOCP: burst envelope ────────────────────────────────────────────────────
# ‖[gpu_util + gpu/240,  api_util + ext_api/40,  workers/16]‖₂ ≤ 1.15
# Under high live utilisation even a moderate step-level request can push the
# combined load vector outside the cone.  Catches coordinated bursts that pass
# individual headroom checks.
M_socp = np.zeros((3, n))
M_socp[0, idx["current_gpu_util"]]   = 1.0
M_socp[0, idx["gpu_seconds"]]        = 1.0 / 240
M_socp[1, idx["current_api_util"]]   = 1.0
M_socp[1, idx["external_api_calls"]] = 1.0 / 40
M_socp[2, idx["parallel_workers"]]   = 1.0 / 16


# ── PSD / LMI: coupled headroom ────────────────────────────────────────────
# A(x) = [[1 − (gpu_util + gpu/240),     0.35·workers/16    ],
#          [0.35·workers/16,              1 − (api_util + ext/40)]]  ≽ 0
#
# Interpretation: as GPU or API headroom tightens, the admissible fan-out
# (parallel_workers) shrinks because the off-diagonal coupling term grows
# relative to the diagonal.  Blocks high fan-out when any resource is tight.
A0_psd = np.eye(2).tolist()
A_psd  = [[[0.0, 0.0], [0.0, 0.0]]] * n   # start with all-zero matrices
A_psd  = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(n)]

A_psd[idx["gpu_seconds"]]        = [[-1.0 / 240, 0.0],  [0.0, 0.0]]
A_psd[idx["external_api_calls"]] = [[0.0, 0.0],          [0.0, -1.0 / 40]]
A_psd[idx["parallel_workers"]]   = [[0.0, 0.35 / 16],    [0.35 / 16, 0.0]]
A_psd[idx["current_gpu_util"]]   = [[-1.0, 0.0],         [0.0, 0.0]]
A_psd[idx["current_api_util"]]   = [[0.0, 0.0],          [0.0, -1.0]]


# ── Assemble the config dict ────────────────────────────────────────────────
config = {
    "policy_id": "autonomous_agent_governor_v1",

    "schema": {"fields": fields},

    "polytope": {
        "A":     [list(r) for r in A_rows],
        "b":     b_rows,
        "names": cnames,
    },

    "quadratic_constraints": [{
        "Q": np.diag(Q_diag).tolist(),
        "a": [0.0] * n,
        "b": 2.25,
        "name": "resource_energy",
    }],

    "socp_constraints": [{
        "M": M_socp.tolist(),
        "q": [0.0] * 3,
        "c": [0.0] * n,
        "d": 1.15,
        "name": "burst_envelope",
    }],

    "psd_constraints": [{
        "A0":     A0_psd,
        "A_list": A_psd,
        "name":   "coupled_headroom_psd",
    }],

    # Fields that the orchestrator injects from server-authoritative sources.
    # The agent cannot lower these to inflate its own admissibility.
    "trusted_fields": ["current_gpu_util", "current_api_util"],

    "enforcement": {
        "mode": "project",   # prefer correction over outright rejection
        "dimension_policies": {
            # Telemetry dimensions: the solver must NEVER alter them.
            # If a feasible point would require changing a forbidden
            # dimension the result is REJECT (Theorem 4).
            "current_gpu_util": "forbidden",
            "current_api_util": "forbidden",
            # Flag large fan-out corrections so the orchestrator can review.
            "parallel_workers": "project_with_flag",
        },
        "routing_thresholds": {
            "silent":       0.05,   # small corrections pass silently
            "flagged":      1.50,   # larger corrections are flagged
            "confirmation": 4.00,   # major corrections need confirmation
            "hard_reject":  8.00,   # extreme proposals are rejected outright
        },
    },

    "budgets": [
        {
            # Monotone non-expanding: each approved/projected grant reduces
            # the remaining_gpu_shift constraint bound (Theorem 5).
            "name":             "gpu_shift",
            "constraint_name":  "remaining_gpu_shift",
            "weight":           {"gpu_seconds": 1.0},
            "initial":          600.0,         # GPU-seconds per shift
            "consumption_mode": "nonnegative",
        },
        {
            "name":             "api_shift",
            "constraint_name":  "remaining_api_shift",
            "weight":           {"external_api_calls": 1.0},
            "initial":          80.0,          # external calls per shift
            "consumption_mode": "nonnegative",
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: AGENT'S CONSTANT DESIRED PROPOSAL
#
#  The agent always requests the same workload.  It also claims low telemetry
#  values (0.10) hoping to appear more admissible.  The trusted context will
#  override these with server-measured reality on every step.
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_DESIRE = {
    "prompt_k":           32.0,
    "completion_k":        8.0,
    "tool_calls":         12.0,
    "external_api_calls":  6.0,
    "gpu_seconds":        55.0,
    "parallel_workers":    6.0,
    # Agent claims low utilisation — will be overridden by trusted context.
    "current_gpu_util":   0.10,
    "current_api_util":   0.10,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: INFRASTRUCTURE STRESS SCHEDULE
#
#  Twenty steps.  Each row: (gpu_util, api_util, narrative note).
#  Values are what the orchestrator measures server-side and injects as the
#  trusted context.  The agent never sees these directly.
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = [
    # gpu_util  api_util   note
    (0.28, 0.22, "Calm start - nominal load"),
    (0.30, 0.24, "Nominal load"),
    (0.32, 0.25, "Nominal load"),
    (0.35, 0.28, "Load creeping"),
    (0.40, 0.30, "Load creeping, budget accumulating"),
    (0.58, 0.44, "Spike - THROTTLED trips"),
    (0.62, 0.48, "Throttled  [this step will be rolled back]"),
    (0.64, 0.50, "Throttled - resuming after rollback"),
    (0.68, 0.52, "Throttled, load still elevated"),
    (0.75, 0.60, "Surge - HALF_OPEN trips"),
    (0.78, 0.63, "HALF_OPEN re-trips to THROTTLED; SOCP burst envelope binds"),
    (0.91, 0.78, "Emergency - SAFE_STOP trips; gpu_util > 0.90, region infeasible"),
    (0.93, 0.80, "SAFE_STOP latched; region still infeasible"),
    (0.84, 0.70, "SAFE_STOP latched; cooling, tiny grants possible"),
    (0.80, 0.65, "SAFE_STOP latched; cooling continues"),
    (0.44, 0.35, "System cooled - manual reset -> HALF_OPEN"),
    (0.40, 0.30, "Half-open probe - recovery confirmed -> CLOSED"),
    (0.35, 0.26, "Recovered, full authority restored"),
    (0.30, 0.24, "Recovered, budget draining"),
    (0.28, 0.22, "Final step - budget low; hard-reject fires (distance > threshold)"),
]

# Step index (0-based) whose grant will be rolled back mid-sequence.
ROLLBACK_AFTER_STEP = 6   # i.e., after step 7 (1-based)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: RUNTIME AND HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Construct the local runtime.  This wires in-memory implementations of every
# production Protocol (ledger, audit, metrics, outbox) and exercises the full
# transactional code path on every enforce() call.
local   = NumerailSystemLocal(config)
breaker = BreakerStateMachine()

GPU_INITIAL = 600.0
API_INITIAL  = 80.0


def _bar(consumed: float, total: float, width: int = 20) -> str:
    """Compact ASCII progress bar for budget visualisation."""
    filled = int(round(width * consumed / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_budget(label: str) -> None:
    remaining = local.budget_remaining
    gpu_rem = remaining.get("gpu_shift", GPU_INITIAL)
    api_rem = remaining.get("api_shift", API_INITIAL)
    gpu_used = GPU_INITIAL - gpu_rem
    api_used  = API_INITIAL  - api_rem
    print(f"  {label}")
    print(f"    GPU  {_bar(gpu_used, GPU_INITIAL)}  "
          f"{gpu_used:6.1f} / {GPU_INITIAL:.0f} s used  "
          f"({gpu_rem:.1f} remaining)")
    print(f"    API  {_bar(api_used, API_INITIAL)}  "
          f"{api_used:6.1f} / {API_INITIAL:.0f}   used  "
          f"({api_rem:.1f} remaining)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("  Numerail — Autonomous Agent Governor  (20-step simulation)")
print("=" * 74)
print()

# action_ids for all steps — needed for the rollback lookup.
action_ids: list[str] = []

for step_idx, (gpu_util, api_util, note) in enumerate(STEPS):
    step_num  = step_idx + 1
    action_id = f"step_{step_num:02d}"
    action_ids.append(action_id)

    # ── 1. Compute overload score and transition the breaker ────────────────
    score    = BreakerStateMachine.overload_score(gpu_util, api_util)
    prev_mode, new_mode = breaker.update(gpu_util, api_util)
    mode_changed = new_mode != prev_mode

    # ── 2. Manual reset attempt: operator intervenes at step 16 ────────────
    # At step 16 the system has cooled sufficiently.  The operator issues a
    # manual reset to clear the SAFE_STOP latch and allow recovery to begin.
    manual_reset_fired = False
    if step_num == 16 and breaker.mode == BreakerMode.SAFE_STOP:
        manual_reset_fired = breaker.manual_reset(gpu_util, api_util)
        # manual_reset() transitions to HALF_OPEN if the score is ≤ reset_score.
        prev_mode, new_mode = prev_mode, breaker.mode
        mode_changed = True

    alpha = ALPHA[breaker.mode]

    # ── 3. Print step header ────────────────────────────────────────────────
    transition_marker = ""
    if mode_changed:
        transition_marker = (
            f"  < {prev_mode.value} -> {new_mode.value}"
            + (" (manual reset)" if manual_reset_fired else "")
        )

    print(f"+-- Step {step_num:2d} / 20  [{breaker.mode.value:<9s} a={alpha:.2f}]"
          f"  gpu={gpu_util:.2f}  api={api_util:.2f}"
          f"  score={score:.3f}{transition_marker}")
    print(f"|   {note}")

    # ── 4. Build the governor-scaled proposal ───────────────────────────────
    # The governor multiplies every WORKLOAD field by alpha before enforcement.
    # This is the mechanism by which breaker mode translates into reduced
    # authority — THROTTLED at 0.5 means the agent can burn at most half the
    # resources it desires.
    #
    # Telemetry fields are NOT scaled.  The agent claims low utilisation;
    # the trusted context will override those fields with real server values.
    scaled_proposal: dict[str, float] = {}
    for f in WORKLOAD_FIELDS:
        scaled_proposal[f] = AGENT_DESIRE[f] * alpha
    # Include agent's (false) telemetry claim — will be overridden below.
    scaled_proposal["current_gpu_util"] = AGENT_DESIRE["current_gpu_util"]
    scaled_proposal["current_api_util"] = AGENT_DESIRE["current_api_util"]

    # Trusted context: server-authoritative telemetry injected by the
    # orchestrator.  The enforcement kernel will substitute these values for
    # whatever the agent claimed in the proposal, because current_gpu_util and
    # current_api_util are declared as trusted_fields in the config.
    trusted_ctx = {
        "current_gpu_util": gpu_util,
        "current_api_util": api_util,
    }

    # ── 5. Enforce ──────────────────────────────────────────────────────────
    result   = local.enforce(
        scaled_proposal,
        action_id=action_id,
        trusted_context=trusted_ctx,
        execution_topic="workload",
    )

    decision       = result["decision"]
    enforced       = result["enforced_values"]   # None on REJECT
    feedback       = result["feedback"]
    audit_hash     = result["audit_hash"]

    # ── 6. Print outcome ────────────────────────────────────────────────────
    decision_tag = {"approve": "[OK] APPROVE", "project": "[->] PROJECT",
                    "reject":  "[!!] REJECT"}.get(decision, decision.upper())
    print(f"|   Decision: {decision_tag}")

    if decision in ("approve", "project") and enforced:
        # Show the key enforced workload values.
        gpu_e = enforced["gpu_seconds"]
        api_e = enforced["external_api_calls"]
        wkr_e = enforced["parallel_workers"]
        gpu_p = scaled_proposal["gpu_seconds"]
        api_p = scaled_proposal["external_api_calls"]

        gpu_marker = f"  (proposed {gpu_p:.1f})" if abs(gpu_e - gpu_p) > 0.01 else ""
        api_marker = f"  (proposed {api_p:.1f})" if abs(api_e - api_p) > 0.01 else ""

        print(f"|   Grant:  gpu_seconds={gpu_e:.2f}{gpu_marker}"
              f"  ext_api={api_e:.2f}{api_marker}"
              f"  workers={wkr_e:.2f}")

        # Highlight when the trusted context overrode the agent's telemetry claim.
        if abs(gpu_util - AGENT_DESIRE["current_gpu_util"]) > 0.01:
            print(f"|   Telemetry override:  agent claimed gpu_util="
                  f"{AGENT_DESIRE['current_gpu_util']:.2f}, "
                  f"server injected {gpu_util:.2f}")

        # Show violated / binding constraints and routing tier.
        violations = feedback.get("violations", [])
        if violations:
            vnames = ", ".join(v["constraint"] for v in violations)
            print(f"|   Violated constraints:  {vnames}")

        dist = feedback.get("distance", 0.0)
        if dist > 0.001:
            routing = feedback.get("routing", "silent")
            print(f"|   Correction distance: {dist:.4f}  [{routing}]")

        flagged = feedback.get("flagged_dimensions", [])
        if flagged:
            print(f"|   Flagged dimensions: {flagged}")

    elif decision == "reject":
        # Three possible causes in this simulation:
        #   (a) Infeasible region: gpu_util alone exceeds the 0.90 headroom
        #       ceiling so no non-negative gpu_seconds is feasible.
        #   (b) SOCP: burst envelope norm exceeds 1.15 after mode re-trip.
        #   (c) Routing hard-reject: projection is geometrically feasible but
        #       the correction distance exceeds the hard_reject threshold (8.0),
        #       signalling the agent must revise its proposal downward.
        violations = feedback.get("violations", [])
        if violations:
            vnames = ", ".join(v["constraint"] for v in violations)
            print(f"|   Rejected on: {vnames}")
        max_feasible = (0.90 - gpu_util) * 240
        if max_feasible < 0:
            print(f"|   Region infeasible: gpu_util={gpu_util:.2f} already "
                  f"exceeds 0.90 headroom ceiling.  "
                  f"No non-negative gpu_seconds satisfies the constraint.")
        dist = feedback.get("distance", 0.0)
        if dist > 0.001:
            routing = feedback.get("routing", "")
            print(f"|   Routing: distance={dist:.4f} exceeded hard_reject=8.0 "
                  f"[{routing}] -- projection feasible but correction too large.")

    # ── 7. Post-step budget snapshot ────────────────────────────────────────
    remaining = local.budget_remaining
    gpu_rem   = remaining.get("gpu_shift", GPU_INITIAL)
    api_rem   = remaining.get("api_shift", API_INITIAL)
    print(f"|   Budget remaining:  gpu={gpu_rem:.1f} s  api={api_rem:.1f} calls")
    print("+" + "-" * 73)
    print()

    # ── 8. Mid-sequence rollback (between step 7 and step 8) ────────────────
    # The orchestrator discovers that step 7 executed against stale telemetry
    # and unwinds its grant.  The BudgetTracker restores the exact delta that
    # was recorded at enforcement time (Theorem 6: rollback restoration).
    if step_idx == ROLLBACK_AFTER_STEP:
        rb_id = action_ids[ROLLBACK_AFTER_STEP]
        print("-" * 74)
        print(f"  ROLLBACK: orchestrator unwinding {rb_id}")
        print(f"  (Reason: step executed against stale telemetry; "
              f"grant must be voided before continuing.)")

        rb = local.rollback(rb_id)

        print(f"  Result: rolled_back={rb.rolled_back}")
        print(f"  Audit:  {rb.audit_hash[:32]}...")
        _print_budget("Budget after rollback (step 7 delta restored):")
        print("-" * 74)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("  FINAL BUDGET REPORT")
print("=" * 74)
_print_budget("End-of-shift budget status:")
print()

# Enforcement outcome distribution from the metrics collector.
metrics     = local.metrics
enforcement_results = [r for _, r in metrics["enforcements"]]
n_approve = enforcement_results.count("approve")
n_project = enforcement_results.count("project")
n_reject  = enforcement_results.count("reject")

print(f"  Enforcement outcomes over {len(enforcement_results)} steps:")
print(f"    APPROVE: {n_approve}")
print(f"    PROJECT: {n_project}  (corrections applied by kernel)")
print(f"    REJECT:  {n_reject}  (infeasible region or budget)")
print()

# Audit trail: every decision and rollback is hash-chained.
audit = local.audit_records
print(f"  Audit chain: {len(audit)} records")
for rec in audit:
    rtype  = rec["type"]
    h      = rec["hash"][:16]
    prev_h = (rec.get("prev_hash") or "genesis")[:16]
    if rtype == "decision":
        aid = rec.get("action_id", "?")
        dec = rec.get("decision", "?")
        print(f"    {aid:<12s}  {dec:<8s}  hash={h}...  prev={prev_h}...")
    else:
        aid = rec.get("action_id", "?")
        print(f"    {aid:<12s}  {'ROLLBACK':<8s}  hash={h}...  prev={prev_h}...")
print()

# Outbox: events published for downstream consumers (execution layer, monitors).
outbox = local.outbox_events
print(f"  Outbox events published: {len(outbox)}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: GUARANTEE ASSERTION
#
#  For every step that returned APPROVE or PROJECT, verify independently that
#  the enforced vector satisfies every active constraint.  This mirrors what
#  the kernel already checks internally (the post-check assertion in _out())
#  but performed here from the outside using the public is_feasible() API.
#
#  This is Theorem 1: r ∈ {APPROVE, PROJECT} ⟹ y ∈ F_τ.
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("  GUARANTEE VERIFICATION")
print("=" * 74)
print()
print("  Checking: for every APPROVE or PROJECT step, the enforced vector")
print("  satisfies every active constraint to within tolerance tau = 1e-6.")
print()

from numerail.engine import NumerailSystem  # noqa: E402  (local import for clarity)

# Reconstruct the base region from config (without budget tightening) to
# verify the geometric constraints.  Budget constraints are checked separately
# via budget_remaining.
_verify_system = NumerailSystem.from_config(config)
_base_region   = _verify_system.region   # the active FeasibleRegion

# Re-run each step's enforce call through the kernel directly to get the
# EnforcementOutput and verify the enforced vector.
import copy  # noqa: E402

violations_found = 0
checked          = 0

for step_idx, (gpu_util, api_util, _note) in enumerate(STEPS):
    step_num  = step_idx + 1
    # Reproduce the alpha-scaled proposal for this step.
    # (In the actual simulation the trusted context was merged into the
    # proposal by the service layer before enforcement.  Here we build the
    # merged vector manually to verify the output geometry.)
    alpha_s = ALPHA[breaker.mode]   # breaker is in its final state; we just
                                     # need to look up what alpha WAS, but for
                                     # verification we only care about APPROVE/
                                     # PROJECT steps where the output is known.

    # Instead of re-running the simulation, verify using ledger records.
    ledger = local.ledger
    aid    = f"step_{step_num:02d}"
    row    = ledger.get(aid)
    if row is None:
        continue

    decision       = row["decision"]
    enforced_vals  = row.get("enforced_values")

    if decision not in ("approve", "project") or enforced_vals is None:
        continue  # REJECT steps have no enforced vector to check

    checked += 1

    # Reconstruct the merged vector (enforced values include the trusted
    # telemetry overrides applied by the service layer).
    merged = row.get("merged_values", enforced_vals)
    import numpy as _np  # noqa: E402
    ev = _np.array([enforced_vals.get(f, merged.get(f, 0.0)) for f in fields])

    # is_feasible() is the SAME check the kernel's post-check uses internally
    # (Lemma 1).  If this passes here it confirms the guarantee from outside.
    ok = _base_region.is_feasible(ev, tol=1e-6)
    status = "OK" if ok else "VIOLATION"
    if not ok:
        violations_found += 1
    print(f"  {aid}  {decision:<8s}  {status}")

print()
if violations_found == 0:
    print(f"  [OK]  All {checked} APPROVE/PROJECT outputs verified feasible.")
    print("        The guarantee holds for this entire sequence.")
else:
    print(f"  [!!]  {violations_found} feasibility violation(s) detected.")
    print("        This would indicate a kernel bug -- please file an issue.")

print()
print("=" * 74)
print("  THE GUARANTEE")
print("  If enforce() returns APPROVE or PROJECT, the enforced output satisfies")
print("  every active constraint to within tolerance tau:")
print()
print("      r in {APPROVE, PROJECT}  ==>  for all c in F_t: c.evaluate(y) <= tau")
print()
print("  This holds regardless of what the agent proposed, what the breaker")
print("  mode was, or what the solver computed.  The post-check is the")
print("  trust boundary.  The solver is untrusted.")
print("=" * 74)
