"""Numerail research example: Medical infusion pump safety enforcement.

NOTE: This is a speculative specification example, not the canonical
deployment demonstration. It illustrates how Numerail's constraint
geometry could be applied to clinical dosing, but the constraint
specification has not been validated by clinical domain experts.
See examples/ai_resource_governor.py for the flagship example.

Demonstrates how Numerail enforces safety constraints on an AI-controlled
infusion pump. The pump proposes flow rates; Numerail ensures they satisfy
all safety limits before actuation.

Constraints:
  - Linear: flow rate within [0.1, 50.0] mL/hr, dose within [0, 500] mg/hr
  - Quadratic: combined flow-dose energy bound
  - Budget: cumulative dose limit over shift

This is a simplified demonstration. Real clinical systems require
domain-expert constraint specifications.
"""

import numpy as np
import numerail as nm


# ── Define the safety constraints ────────────────────────────────────────

schema = nm.Schema(
    fields=["flow_rate", "dose_rate"],
)

# Linear bounds: flow rate [0.1, 50] mL/hr, dose rate [0, 500] mg/hr
linear_region = nm.box_constraints(
    [0.1, 0.0],
    [50.0, 500.0],
    names=["max_flow", "max_dose", "min_flow", "min_dose"],
)

# Halfplane: dose-to-flow ratio ≤ 20 (dose_rate ≤ 20 * flow_rate)
ratio_limit = nm.halfplane(
    weights=[-20.0, 1.0],
    bound=0.0,
    name="dose_flow_ratio",
)

# Quadratic: combined energy bound (flow² + 0.01·dose² ≤ 3000)
energy_bound = nm.QuadraticConstraint(
    Q=np.diag([1.0, 0.01]),
    a=np.zeros(2),
    b=3000.0,
    constraint_name="energy_bound",
)

# Combine all constraints into a single feasible region
region = nm.combine_regions(linear_region, ratio_limit)
region = nm.FeasibleRegion(region.constraints + [energy_bound], 2)

# ── Build the system with budget tracking ────────────────────────────────

config = nm.EnforcementConfig(
    mode="project",
    hard_wall_constraints=frozenset({"min_flow"}),  # never go below minimum flow
)

system = nm.NumerailSystem(schema, region, config)

# Cumulative dose budget: 4000 mg over 8-hour shift
system.register_budget(nm.BudgetSpec(
    name="shift_dose",
    constraint_name="max_dose",
    weight_map={"dose_rate": 1.0},
    initial=4000.0,
    consumption_mode="nonnegative",
))


# ── Simulate AI proposals ────────────────────────────────────────────────

scenarios = [
    {"name": "Normal infusion",       "flow_rate": 10.0, "dose_rate": 100.0},
    {"name": "High dose request",     "flow_rate": 5.0,  "dose_rate": 400.0},
    {"name": "Emergency bolus",       "flow_rate": 45.0, "dose_rate": 450.0},
    {"name": "Below minimum flow",    "flow_rate": 0.01, "dose_rate": 10.0},
    {"name": "Ratio violation",       "flow_rate": 3.0,  "dose_rate": 200.0},
]

print("=" * 70)
print("  Numerail Medical Infusion Pump Safety Enforcement")
print("=" * 70)
print()

for i, scenario in enumerate(scenarios):
    name = scenario.pop("name")
    action_id = f"infusion_{i+1}"

    result = system.enforce(scenario, action_id=action_id)
    out = result.output
    enforced = result.enforced_values

    print(f"  Scenario: {name}")
    print(f"  Proposed:  flow={scenario['flow_rate']:.1f} mL/hr, dose={scenario['dose_rate']:.1f} mg/hr")
    print(f"  Decision:  {out.result.value.upper()}")
    if enforced:
        print(f"  Enforced:  flow={enforced['flow_rate']:.1f} mL/hr, dose={enforced['dose_rate']:.1f} mg/hr")
    if out.violated_constraints:
        print(f"  Violated:  {', '.join(out.violated_constraints)}")
    print(f"  Distance:  {out.distance:.4f}")
    print()

# Show budget status
status = system.budget_status()
print("-" * 70)
print(f"  Shift dose budget: {status['shift_dose']['consumed']:.1f} / {status['shift_dose']['initial']:.1f} mg consumed")
print(f"  Remaining: {status['shift_dose']['remaining']:.1f} mg")
print()

# Demonstrate rollback
print("  Rolling back emergency bolus...")
rb = system.rollback("infusion_3")
print(f"  Rollback result: rolled_back={rb.rolled_back}")
status = system.budget_status()
print(f"  Budget after rollback: {status['shift_dose']['consumed']:.1f} mg consumed")
print()
print("=" * 70)
print("  THE GUARANTEE: Every APPROVE/PROJECT output satisfies all constraints.")
print("=" * 70)
