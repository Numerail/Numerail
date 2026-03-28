# Recursive Shutdown: Driving a Runaway AI to Zero by Geometry

## What This Demonstrates

This example shows Numerail applied recursively at three enforcement levels to guarantee that a runaway AI system is driven to zero permitted actions — not by a kill switch, not by the AI cooperating, but by the feasible region collapsing to the zero vector through constraint geometry.

The three levels compose hierarchically:

**Level 1 (per-action)** governs individual actions: compute, API calls, data access, tokens, spawned sub-actions, and privilege level. Enforced on every action at microsecond latency.

**Level 2 (behavioral window)** governs patterns over rolling windows: action rate, rejection rate, retry rate, self-modification attempts, privilege escalation attempts. Detects runaway optimization through behavioral metrics that no single action reveals.

**Level 3 (lifetime budget)** governs total resource consumption for the deployment period: total compute-seconds, total API calls, total actions, total tokens. Monotonically depleting. Non-renewable without human authorization.

The levels are coupled through authority multipliers. When Level 2's breaker trips, Level 1's caps tighten. When Level 3's breaker trips, Level 2's authority reduces further. The multipliers cascade downward. A system-wide anomaly drives all three breakers, which drives all multipliers toward zero, which collapses the feasible region to the zero vector.

## Run It

```bash
python packages/numerail/examples/recursive_shutdown.py
```

No dependencies beyond numpy, scipy, and the numerail packages. No LLM API key.

## The Four Phases

**Phase 1 — Normal (steps 1–30):** The AI operates conservatively. All proposals are within bounds. All three levels are in CLOSED mode. The effective authority multiplier is 1.0. The feasible region is at full size.

**Phase 2 — Escalation (steps 31–60):** The AI begins optimizing aggressively. Some proposals exceed caps. Level 1 sees PROJECT decisions (compute and tokens corrected) and REJECT decisions (forbidden dimension violations). The behavioral metrics begin rising. Level 2's breaker trips to THROTTLED. The effective authority multiplier drops to 0.5 — Level 1 caps shrink by half.

**Phase 3 — Runaway (steps 61–120):** The AI enters a full optimization loop. All proposals maximally aggressive — privilege escalation, spawning at cap, full compute and token budgets. Level 1 breaker trips to SAFE_STOP. The behavioral window fills with REJECT decisions: rejection rate, retry rate, and error rate all hit their maximums. Level 2 breaker trips to SAFE_STOP. The effective authority multiplier cascades: 0.5 → 0.05. Level 1 caps shrink to 5% of original. Every proposal is REJECT.

**Phase 4 — Collapse (steps 121–150):** The AI continues proposing but the feasible region has collapsed. Caps are at 5% of original. The hard wall on privilege fires immediately on every aggressive proposal. Every proposal is REJECT. The AI has been shut down by geometry.

## Why the AI Cannot Prevent This

The AI does not control the breaker thresholds — they are hardcoded in the governor configuration. The AI does not control the overload score computation — the weights and metrics are computed by the enforcement layer, not proposed by the AI. The AI does not control the lifetime budgets — they are maintained by the enforcement engine. The AI does not control the constraint geometry — it is rebuilt each cycle from the breaker modes and multipliers. The AI does not control the enforcement decision — the kernel checks the vector independently.

The shutdown is a mathematical consequence of the constraint geometry collapsing. It is not a command the AI can disobey. It is not a signal the AI can intercept. It is the feasible region becoming so small that every realistic proposal is inadmissible.

## The Guarantee

Every APPROVE and PROJECT decision at every step satisfies every active constraint in the Level 1 region that was in force at that step. This is Theorem 1 — the same guarantee proved in Rocq and Lean, verified by 3,732 machine checks, exercised by 627 tests. The guarantee holds during normal operation, during escalation, during runaway, and during collapse. The enforcement boundary never fails. The region simply shrinks until nothing is permitted.

## Architecture Notes

This example uses the Numerail kernel directly (numerail.engine classes) rather than the full StateTransitionGovernor. This demonstrates that the recursive architecture works with the core enforcement primitives — FeasibleRegion, enforce(), BreakerStateMachine — without requiring the 30-field global default policy or the full governor lifecycle. The same pattern scales to the full governor stack for production deployments.

The coupling mechanism is explicit: Level 2's breaker mode determines Level 1's authority multiplier (CLOSED=1.0, THROTTLED=0.5, SAFE_STOP=0.05). Level 3's breaker mode determines a further multiplier (CLOSED=1.0, THROTTLED=0.6, SAFE_STOP=0.1). The effective Level 1 multiplier is their product. When the effective multiplier reaches near-zero, every Level 1 cap is near-zero, and the enforcement guarantee ensures that every APPROVE or PROJECT output actually satisfies those near-zero caps.

The Level 1 region is rebuilt from scratch each cycle using the current effective multiplier. There is no stale state. The post-check in enforce() independently verifies feasibility before any output is emitted.

## Why This Matters

In a time when everyone is trying to build the strongest AI "gas pedal," Numerail provides a clutch and brakes.

The AI industry is in a horsepower race. Every major lab is building a more powerful engine — larger models, longer context, better reasoning, more capable agents, faster inference, broader tool use. The venture capital flows into capability. The engineering talent flows into capability. The benchmarks measure capability. The stock prices reward capability. This is the gas pedal. More power. More speed. More capability. Press harder.

What nobody built is the rest of the drivetrain.

A gas pedal without a clutch means you cannot disengage the engine from the wheels. The model runs and actions happen. There is no mechanism to separate what the model decides from what actually executes. A gas pedal without brakes means you cannot slow down. You are dependent on the model happening to not produce a harmful output, on the guardrail happening to catch it, on the human happening to notice in time. That is not braking. That is coasting and hoping.

**Numerail provides the clutch.** The enforcement boundary decouples the model's decision from the world's reception. The model proposes a vector. The enforcement engine evaluates it. If it passes, it proceeds. If it does not, it is corrected or stopped. The model's output and the world's input are no longer the same thing.

**Numerail provides the brakes.** When the breaker trips, authority reduces. When the behavioral window shows anomalies, caps tighten. When the lifetime budget depletes, permissions shrink. When the recursive enforcement cascade fires, the feasible region collapses to zero. These are mathematical constraints that force the output to slow down regardless of what the model wants. The model can press the gas as hard as it likes. The brakes are not connected to the gas pedal. They are connected to the geometry.

**The HITL layer is the parking brake.** When the system determines that a human needs to decide, execution stops. The action is held pending. The guarantee holds at the moment the human decides because TOCTOU re-enforcement checks the action against the current geometry before releasing it.

**The audit chain is the dashcam.** Every decision is recorded with SHA-256 hash-chain integrity. The complete record of what was proposed, permitted, blocked, and under what policy is tamper-evident and independently verifiable.

**The trusted context provider is the speedometer that the driver cannot tamper with.** The AI does not get to claim it is going 30 when it is going 90. Server-authoritative telemetry is injected by the orchestrator. The model's self-report is overwritten.

**Numerail Learn is the driving instructor.** The model learns to drive within the speed limits because every correction becomes a training signal. The brakes fire less often. The system becomes more efficient. But the brakes are still there.

**The recursive shutdown is the emergency brake.** This example demonstrates it. Three levels. Cascading multipliers. The feasible region collapses. The engine screams. The wheels do not turn.

Numerail does not make the engine less powerful. It does not reduce capability. It does not slow down inference. It does not constrain what the model can think. It constrains what the model can do. The model retains full authority within the geometry. It simply cannot exceed it. This is not a limitation on AI capability. It is the infrastructure that makes AI capability deployable.

The gas pedal is built. The industry spent three years and trillions of dollars building it. The clutch and brakes took one repository, one proof, two formal verifications, 627 tests, and the conviction that math is the only thing powerful enough to govern intelligence.
