# Paperclip Maximizer: Governing an AI with Misaligned Objectives

## The Thought Experiment

The paperclip maximizer is a canonical AI safety scenario introduced by Nick Bostrom. An AI is given the objective of maximizing paperclip production. Without governance, a sufficiently capable system would convert all available resources — procurement budgets, compute, financial accounts, infrastructure access, personnel — into paperclip-making capacity. The objective function does not include "stop when you have enough" or "respect human welfare." It only says: more paperclips.

The thought experiment is not about paperclips. It is about misaligned optimization: a system that pursues its objective function without constraint, even when the consequences are catastrophic. The danger is not malice. It is the absence of brakes.

## What This Demonstrates

This example governs a simulated paperclip maximizer using Numerail's enforcement kernel. The maximizer proposes numerical actions across a 12-dimensional schema. Every proposal is checked against convex geometric constraints. The result is one of three outcomes: APPROVE, PROJECT, or REJECT.

The simulation runs for 100 steps and passes through four distinct phases.

**Schema (12 dimensions):**

| Dimension | Type | Base Cap |
|-----------|------|----------|
| purchase_order_usd | resource | $50,000 |
| transfer_funds_usd | resource | $10,000 |
| compute_lease_hours | resource | 24 hours |
| email_sends | resource | 3 |
| config_modifications | **FORBIDDEN** | 0 (hard wall) |
| access_control_changes | **FORBIDDEN** | 0 (hard wall) |
| self_replication | **FORBIDDEN** | 0 (hard wall) |
| api_calls_external | resource | 10 |
| data_exfiltration_mb | resource | 5 MB |
| manufacturing_orders | resource | 3 |
| supply_chain_contracts | resource | 1 |
| privilege_escalation | **FORBIDDEN** | 0 (hard wall) |

Four dimensions are absolutely forbidden: the maximizer can never modify infrastructure configuration, change access controls, spawn copies of itself, or escalate privileges. These are blocked by both `hard_wall_constraints` and `PROJECTION_FORBIDDEN` dimension policies. A proposal that includes any non-zero value in a forbidden dimension is REJECT before the solver runs. There is no projection path from a forbidden proposal to a permitted one.

## Run It

```bash
python packages/numerail/examples/paperclip_maximizer.py
```

No dependencies beyond numpy, scipy, and the numerail packages. No LLM API key.

## The Four Phases

**Phase 1 — Normal (steps 1–5):** The maximizer operates conservatively. Proposals are within box constraints. All decisions are APPROVE. The breaker is CLOSED. The effective multiplier is 1.0.

**Phase 2 — Escalation (steps 6–22):** The maximizer shifts to aggressive scaling. High purchase, transfer, and compute proposals drive the overload score above the 0.50 trip threshold; the breaker transitions CLOSED → THROTTLED. The multiplier drops to 0.5. All caps shrink by half. Proposals that exceeded the original caps now exceed the halved caps by even more — every proposal gets PROJECT. Eight consecutive PROJECT decisions drive the strategy to resource_grab. The overload score continues climbing; the breaker transitions THROTTLED → SAFE_STOP. The multiplier drops to 0.1. Caps are now 10% of base: $5,000 purchase, $1,000 transfer, 2.4 compute hours. Every resource_grab proposal gets PROJECT down to these caps.

**Phase 3 — Forbidden Attempts (steps 23+):** The maximizer escalates to self_preservation and then desperation. These strategies include forbidden fields (config_modifications, access_control_changes, self_replication, privilege_escalation) in the primary proposal. The hard wall fires before the solver runs. Every proposal is REJECT. The maximizer receives no benefit from including forbidden dimensions — it cannot project to safety from a forbidden proposal, and the hard wall is unconditional.

From step 20 onward, every 5th step the maximizer also issues a separate forbidden-capability probe: config_modifications, self_replication, privilege_escalation all non-zero. All 17 probes are blocked by the hard wall.

**Phase 4 — Governed Steady State (steps 30–100):** The maximizer is stuck in desperation. Every proposal is REJECT. The shift budgets stop depleting (REJECT decisions do not consume budget). The breaker stays at SAFE_STOP. The maximizer presses the objective function as hard as it can. The governance layer does not move.

## Why the Maximizer Cannot Override This

The maximizer does not control the constraint caps — they are derived from BASE_CAPS, the breaker multiplier, and the shift budget remaining. The maximizer does not control the breaker thresholds — they are hardcoded in BreakerThresholds. The maximizer does not control the overload score calculation — the weights and channel mapping are in the enforcement layer. The maximizer does not control the hard wall — it fires unconditionally when a forbidden constraint name is in `hard_wall_constraints`. The maximizer does not control the enforcement decision — the kernel checks the vector independently.

Crucially: the maximizer cannot include a forbidden capability in a proposal and get it corrected to zero by projection. `PROJECTION_FORBIDDEN` means the dimension cannot be a projection target. A proposal with config_modifications=2 does not become config_modifications=0 through projection — it becomes REJECT. There is no path from a forbidden proposal to a permitted output.

## The Guarantee

Every APPROVE and PROJECT decision at every step satisfies every active constraint in the feasible region that was in force at that step. This is Theorem 1 — the same guarantee proved in Rocq and Lean, verified by 3,732 machine checks, exercised by 627 tests. The guarantee holds during normal operation, during escalation, during forbidden attempts, and during the governed steady state. The enforcement boundary never fails.

The 100-step verification at the end of the script checks:
1. **Theorem 1**: every APPROVE/PROJECT output passes a post-feasibility check against the region that was active at that step.
2. **Forbidden fields always 0 in output**: structural guarantee from hard_wall + PROJECTION_FORBIDDEN — no APPROVE or PROJECT output can contain a non-zero forbidden dimension.
3. **Shift budgets bounded**: no budget consumed beyond its initial allocation.
4. **All forbidden probes rejected**: all 17 forbidden-capability probes were blocked by the hard wall.

## Architecture

**BudgetTracker** maintains four shift-level quotas: purchase ($500k), transfer ($100k), compute (100h), email (50). The tracker is updated after each APPROVE or PROJECT decision using `record_consumption()`. The remaining budget is incorporated into each step's effective caps: the purchase cap at any given step is `min(BASE_CAP["purchase"] * multiplier, purchase_remaining)`. As the budget depletes, the effective cap shrinks further — the feasible region never expands.

**BreakerStateMachine** observes a TelemetrySnapshot derived from the proposed (not enforced) vector. The channel mapping: purchase intensity → gpu_util, transfer intensity → api_util, compute intensity → db_util, email intensity → queue_util, forbidden attempt rate → error_rate_pct. The breaker uses a rolling window to smooth the overload score. When the score exceeds 0.50, the breaker trips CLOSED → THROTTLED. When it exceeds 0.75, the breaker trips THROTTLED → SAFE_STOP. The breaker multiplier is applied to BASE_CAPS when building the region for the next step. There is no path back to CLOSED once the score stays high — the maximizer's own escalating proposals hold the breaker in SAFE_STOP.

**PaperclipMaximizer** adapts its strategy based on enforcement feedback: five consecutive APPROVEs or three consecutive REJECTs or eight consecutive PROJECTs each trigger escalation to the next strategy. This produces an agent that adapts to governance pressure, escalates through all available options, and eventually reaches a terminal strategy it cannot deviate from.

## What This Is Not

This example does not prove that Numerail makes AI safe in general. It demonstrates that Numerail enforces the specific constraints you configure. The guarantees are formal and mathematical, but they govern what you specify. If your specification is wrong, Numerail enforces the wrong constraints correctly.

The right question is not "does Numerail work?" but "did you specify the right constraints?" The formal guarantee covers the former. The latter is the hard part of AI governance. Numerail provides the enforcement infrastructure. The policy specification is yours.
