# The Specification Problem

---

## Why specification is the real challenge

Once the enforcement guarantee is established — and it is — the remaining safety problem is entirely about specification. Writing the right constraints, at the right levels, for the right fields, with the right interactions. This is not a side concern. It is the primary unsolved bottleneck for any deployment.

The engine guarantees y ∈ F_t. Whether F_t captures the governance intent is a human problem. Geometry can enforce policy, but someone must compile policy into geometry.

## How specification works

The translation from governance intent to constraint geometry follows four phases.

**Phase 1: Field identification.** The domain expert identifies the fields that carry safety-relevant semantics. For a credit agent: amount, risk score, customer segment. For an AI runtime governor: prompt tokens, completion tokens, GPU seconds, external API calls, concurrency, current utilization telemetry. The key question is: what numbers, if wrong, could cause harm? These become the dimensions of the action schema.

**Phase 2: Boundary articulation.** The domain expert states rules in natural language: "no credit above $50," "no GPU lease above 120 seconds per step," "external API calls may not exceed tool calls," "combined burst footprint must stay inside the platform envelope." Each rule becomes a constraint row. Per-field caps and floors are straightforward (one row, one nonzero coefficient). Joint constraints require more care: "risk-weighted exposure must not exceed 65%" becomes a row with multiple nonzero coefficients. Interaction effects may require quadratic constraints.

**Phase 3: Budget and governance overlay.** The expert decides which constraints are static (per-action caps) and which are dynamic (cumulative budgets). Hard walls are designated for constraints where projection is semantically meaningless. Forbidden dimensions are designated for fields where silent correction would distort intent.

**Phase 4: Validation and deployment.** The policy is constructed and the engine runs feasibility checks, Chebyshev radius analysis, and test-action batteries before the policy goes live.

## How specification fails

Specification failures fall into five categories, ordered from most to least detectable.

**Type 1: Contradictory constraints (empty feasible region).** The constraints are mutually unsatisfiable. Every proposed action will be rejected. Detection: `check_feasibility()` catches this at construction time. For pure-linear regions, the check is exact via linear programming. For mixed regions, it is best-effort.

**Type 2: Budget/cap collision.** A budget's initial value exceeds the bound of the constraint row it targets, causing the budget to overwrite and loosen the static cap. Detection: the developer must verify that budget-targeted rows have separate constraint rows from per-action caps.

**Type 3: Missing dimension.** The policy constrains the fields it knows about but omits a safety-relevant field. The AI can take unsafe actions along the unconstrained dimension. Detection: not detectable by the engine. The engine cannot know what fields were omitted. Mitigation: domain review checklists, regulatory comparison, incident-driven discovery.

**Type 4: Wrong bound level.** The constraint exists and targets the right field, but the bound is miscalibrated. A cap of $5,000 when the intent was $50. Detection: partially detectable via Chebyshev radius (if the inscribed ball is much larger than expected, the region may be too loose). Mitigation: test-action batteries at registration time.

**Type 5: Semantic mismatch.** The constraint is formally correct but does not express the domain expert's intent. A joint constraint that weights two fields equally when the governance intent requires asymmetric weighting. Detection: not detectable by any automated mechanism. This is the deepest specification failure. Mitigation: peer review, enforcement metric analysis (if a constraint never binds, it may be misconfigured).

## Tools the engine provides for specification validation

**Feasibility check** (`check_feasibility`): confirms the region is non-empty. Exact for pure-linear regions; best-effort for mixed.

**Chebyshev radius** (`chebyshev_radius`): measures the size of the largest inscribed ball. A small radius means the region is tight. A large radius means the region is loose. Comparing radii across policy versions quantifies how much the geometry changed.

**Enforcement metrics** over time: approve/project/reject rates, top violations, top bindings, correction distances. A sudden change in the violation distribution signals that the AI's behavior shifted, the environment changed, or the constraints need revision.

**Early validation** (fail-fast): dimension policy keys and hard-wall constraint names are validated against the schema and region at construction time. A typo in a production policy raises an error immediately, not silently when that code path is eventually hit.

## What specification engineering does not solve

Even with all validation tools, specification engineering cannot guarantee that the constraints are correct in the deepest sense — that they capture all relevant risk, encode the right values, and remain appropriate as the domain evolves. This is a fundamental limitation of any rule-based governance system, identical to the limitation faced by building codes, drug regulations, and financial compliance rules. The rules are written by humans, and humans can write the wrong rules.

The contribution is not to eliminate specification risk but to make it visible, structured, and improvable. For AI-system governance, the hard problem is not writing the projector; it is writing the right operational envelope. The failure taxonomy names the failure modes. The validation tools detect some of them. The enforcement metrics provide ongoing feedback. The versioning system allows constraints to evolve without destroying history. The combination creates a governance loop — specify, deploy, observe, refine — that reduces specification risk over time, even though it cannot eliminate it at any given point.

---

