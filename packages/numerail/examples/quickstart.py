"""Numerail quickstart — 10 lines to enforce your first constraint."""

import numpy as np
import numerail as nm

# Define a feasible region: all values between 0 and 1
region = nm.box_constraints([0, 0], [1, 1])

# An AI proposes a value outside the region
proposed = np.array([1.5, 0.5])

# Numerail enforces the constraint
output = nm.enforce(proposed, region)

print(f"Result:   {output.result.value}")       # "project"
print(f"Original: {proposed}")                   # [1.5, 0.5]
print(f"Enforced: {output.enforced_vector}")     # [1.0, 0.5]
print(f"Distance: {output.distance:.4f}")        # 0.5000

# THE GUARANTEE: if result is APPROVE or PROJECT,
# the enforced vector satisfies every constraint.
assert region.is_feasible(output.enforced_vector)
