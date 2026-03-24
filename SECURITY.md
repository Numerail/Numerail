# Security Policy

## The enforcement guarantee

Numerail's core safety property is Theorem 1: if `enforce()` returns APPROVE or PROJECT, the output satisfies every active constraint. A vulnerability in this context is any input, configuration, or code path that violates this guarantee — causing the engine to emit an APPROVE or PROJECT for a vector that does not satisfy the active constraints.

## Reporting a vulnerability

If you discover a potential violation of the enforcement guarantee or any other security issue, please report it privately.

**Email:** trynumerail@gmail.com

**Do not** open a public GitHub issue for security vulnerabilities. Responsible disclosure gives us time to verify and fix the issue before it is publicly known.

## What to include

- A description of the vulnerability.
- Steps to reproduce it, or a minimal test case.
- Which theorems or guarantees are affected.
- The version of Numerail where you observed the issue.

## Response

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation within 7 days for confirmed guarantee violations.

## Scope

The following are in scope:

- Violations of Theorem 1 (enforcement soundness) — any code path that emits APPROVE or PROJECT for an infeasible vector.
- Violations of Theorems 2–9 (fail-closed, hard-wall dominance, forbidden-dimension safety, budget monotonicity, rollback restoration, audit integrity, passthrough, idempotence).
- Bypass of trusted context injection (agent-supplied values surviving the merge for declared trusted fields).
- Bypass of scope enforcement (operations succeeding without required scopes).
- Tampering with the audit chain that is not detected by `verify()`.
- Digest collisions or chain linkage bypasses in `NumerailPolicyContract`.

The following are out of scope:

- Specification errors (wrong constraints, wrong bounds). The engine enforces whatever geometry is defined. Misconfigured constraints are a policy problem, not a security vulnerability.
- Denial of service via computationally expensive constraint compositions. The solver chain has configurable iteration limits.
- Dependencies (numpy, scipy). Report those to the respective projects.
