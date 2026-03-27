# Regulatory Bodies and Numerail

## The Problem Regulators Face

Regulatory bodies have spent decades developing frameworks for governing high-risk systems — aviation, medical devices, financial instruments, nuclear facilities. These frameworks share a common structure: define what is permissible, require the operator to demonstrate compliance, and audit the record after the fact. The frameworks work because the systems they govern are deterministic. A bridge either meets the load specification or it does not. A drug either passes the trial protocol or it does not. The regulator defines the specification. The operator builds to it. The auditor verifies it.

AI breaks this model. The systems are probabilistic, the outputs are non-deterministic, and the behavior changes with every input. A regulator cannot write a specification that says "the model shall not hallucinate" because hallucination is a statistical property, not a binary condition. A regulator cannot audit a conversation log and determine whether the model "complied" because compliance is not a property of text — it is a property of actions and their consequences.

Numerail restores the regulatory model by shifting the compliance boundary from the model to the actuation layer. The regulator does not need to specify how the model should behave. The regulator specifies what the model is allowed to do — expressed as convex geometric constraints on the numerical outputs that become real-world actions. The enforcement engine guarantees that every approved action satisfies every constraint. The audit chain provides a tamper-evident record of every action, every constraint evaluation, and every human decision. The policy contract system provides cryptographic proof of which constraints were in force at any point in time.

This means a regulator can define governance requirements as constraint geometry, and any organization deploying AI agents can enforce those requirements with mathematical certainty by loading the regulator's policy into Numerail.

---

## Two Distribution Models

Regulatory bodies can distribute their requirements through Numerail in two ways. Both are complementary. Most regulatory bodies would use both.

### Model 1: Published Policy Packs

A regulatory body publishes a Numerail policy pack — a versioned, digitally digested (SHA-256) configuration file that defines the constraint geometry for a specific regulatory domain. The policy pack is a JSON or YAML document that can be loaded directly by `NumerailSystem.from_config()` or wrapped in a `NumerailPolicyContract` for provenance tracking.

The policy pack contains:

- **Schema definition.** The named dimensions that the regulation governs. For a financial trading regulation, this might include `trade_size_usd`, `position_exposure_pct`, `leverage_ratio`, `order_rate_per_minute`, `counterparty_count`. For an AI agent safety regulation, this might include `api_calls_external`, `data_volume_kb`, `credential_accesses`, `shell_executions`, `email_sends`.

- **Constraint geometry.** The linear, quadratic, SOCP, and PSD constraints that encode the regulatory requirements. A financial regulation might specify: `trade_size_usd ≤ 500000` (per-trade cap), `position_exposure_pct ≤ 0.10` (portfolio exposure limit), `‖[trade_size_usd/500000, leverage_ratio/10]‖ ≤ 1.0` (joint risk norm), and a PSD constraint coupling the interaction between leverage and counterparty concentration. An AI safety regulation might specify per-action caps, shift budgets, joint communication constraints, and energy bounds on combined sensitive operations.

- **Trusted field declarations.** Which fields must come from server-authoritative sources rather than the AI. A financial regulation would declare `current_portfolio_value`, `current_exposure`, `market_volatility_index` as trusted. An AI safety regulation would declare `current_time_ms`, `breaker_mode`, `budget_remaining`, `error_rate` as trusted.

- **Dimension policies.** Which dimensions require PROJECTION_FORBIDDEN (the AI must propose a compliant value rather than having it silently corrected). A financial regulation might require that `trade_size_usd` is never silently projected — if the AI proposes an oversized trade, it must be rejected, not quietly reduced.

- **Enforcement mode and routing thresholds.** Whether the regulation requires reject-only enforcement (no silent corrections), project mode (corrections allowed), or hybrid. The routing thresholds define which corrections are flagged and which require human confirmation.

- **Budget specifications.** If the regulation imposes resource consumption limits per period (daily trading volume caps, monthly API call limits, quarterly data transfer budgets).

- **Metadata.** The regulatory body's identifier, the regulation version, the effective date, the expiration date (if any), the applicable jurisdiction, and references to the source regulation (e.g., "EU AI Act Article 9(4)" or "NIST AI RMF Function: Govern").

The policy pack is distributed as a file — downloadable from the regulatory body's website, published to a package registry, or distributed through a compliance vendor. The deployer loads it into their Numerail instance. The enforcement engine enforces it. The policy contract system records the digest of the loaded policy, creating a cryptographic proof that the deployer was using the regulator's exact specification at the time of each enforcement decision.

**Example: NIST publishes an AI agent safety policy pack**

NIST develops a policy pack for autonomous AI agents operating in critical infrastructure. The pack defines 25 dimensions covering resource consumption, external communication, data access, and operational impact. It includes 60 linear constraints (per-action caps and shift budgets), 2 quadratic constraints (energy bounds on combined sensitive operations), 1 SOCP constraint (joint external communication norm), and 15 trusted fields. The enforcement mode is hybrid with `max_distance = 2.0` — corrections within a small radius are applied silently; larger deviations require human review. The pack references NIST AI RMF functions (Govern, Map, Measure, Manage) and maps each constraint to the specific RMF sub-function it satisfies.

An enterprise deploying an AI agent for cloud infrastructure management downloads the NIST policy pack, loads it into their Numerail instance, and every action the agent takes is enforced against NIST's constraint geometry with a mathematical guarantee. The enterprise's audit trail contains the NIST policy pack's digest at every enforcement decision, proving compliance at every point in time.

**Example: ISO publishes a domain-specific policy pack**

ISO develops a policy pack for AI agents operating in financial services, based on ISO/IEC 42001 (AI management system) and ISO 31000 (risk management). The pack defines 30 dimensions covering trading operations, client data access, reporting obligations, and market interaction. It includes constraints derived from the standard's risk treatment requirements, with each constraint annotated with the specific ISO clause it implements. The pack is versioned and updated when the standard is revised. Financial institutions load the pack and can demonstrate to auditors that their AI trading agents are enforced against the ISO specification with cryptographic proof of policy provenance.

### Model 2: Hosted Enforcement API

A regulatory body operates a Numerail enforcement service as a public API. Organizations send their AI agents' proposed actions to the API and receive enforcement decisions (APPROVE, PROJECT, REJECT) in response. The regulatory body maintains the constraint geometry, updates it as regulations evolve, and provides the audit trail as a service.

The API endpoint accepts:

- The proposed action vector (the numerical values the AI agent wants to execute).
- The deployer's identifier (for audit trail attribution).
- Optional trusted context values (the deployer's infrastructure telemetry, which the API can cross-validate against its own monitoring if integrated).

The API returns:

- The enforcement decision (APPROVE, PROJECT, REJECT).
- The enforced vector (if PROJECT, the corrected values).
- The constraint violations (if REJECT, which constraints were violated and by how much).
- The audit hash (linking this decision to the tamper-evident audit chain).
- The policy digest (proving which constraint geometry was used).

The deployer does not need to run Numerail locally. They do not need to download or manage policy packs. They call the API before every consequential action, and the regulatory body's enforcement boundary is applied with the same mathematical guarantee as a local deployment.

**Advantages of hosted enforcement:**

The regulatory body controls the policy and can update it without requiring deployers to download new versions. The regulatory body maintains the audit trail centrally, enabling cross-organization analysis (with appropriate data governance). The deployer cannot modify the constraint geometry — the enforcement is truly independent of the regulated entity. Compliance is real-time and verifiable: the regulatory body can confirm at any moment whether a specific deployer's last N actions were all APPROVE or PROJECT against the current policy.

**Considerations for hosted enforcement:**

Latency. The API call adds network round-trip time (typically 5-50ms depending on geography) on top of the enforcement computation (25 microseconds for APPROVE). For latency-critical applications, a local deployment with periodic policy sync may be preferable.

Availability. If the API is unavailable, the deployer's AI agent cannot execute any action (fail-closed by default). The deployer needs a fallback strategy — either a cached local policy pack or a fail-safe operational mode.

Data sensitivity. The proposed action vector is sent to the regulatory body's API. If the action vector contains competitively sensitive information (trade sizes, resource allocations, operational details), the deployer may prefer local enforcement with policy packs rather than sending action data to an external service. The API can be designed to accept only the numerical vector without identifying metadata, mitigating but not eliminating this concern. Importantly, the action vector contains only numerical magnitudes (counts, rates, ratios) — not the content of the action itself. The regulator sees "email_sends=1, data_volume_kb=45" but never the email body. This architectural property significantly reduces the data sensitivity concern.

Sovereignty. Some jurisdictions may require that enforcement decisions be computed within the jurisdiction's borders. The hosted API must be deployable in multiple regions.

**Example: A financial regulator operates a hosted enforcement API**

A national financial regulatory authority operates a Numerail API that enforces position limits, trading velocity constraints, and leverage caps for algorithmic trading systems. Every algorithmic trading firm in the jurisdiction is required to route its AI-generated trade proposals through the API before execution. The regulator's API enforces the same constraint geometry for all firms simultaneously, updates constraints in real time in response to market conditions (tightening position limits during volatility events), and maintains a central audit trail of every trade proposal and enforcement decision across the entire market. When a firm's AI trading agent proposes an oversized position, the API projects it to the regulatory limit or rejects it — before the trade reaches the exchange.

---

## How Specific Regulatory Frameworks Map to Numerail

### NIST AI Risk Management Framework (AI RMF)

The NIST AI RMF defines four core functions: Govern, Map, Measure, and Manage. Each maps to Numerail capabilities:

**Govern** (establish and maintain AI risk management processes): The policy contract system provides versioned, cryptographically digested constraint specifications that encode governance requirements. The chain-linked contract history provides an auditable record of every policy change. The HITL layer enforces that high-stakes decisions require human authorization with authenticated reviewer identity.

**Map** (identify and assess AI risks): The schema definition process — determining which dimensions to constrain and how they interact — is the Map function applied to the actuation layer. The global default policy provides a reference mapping for AI agent resource governance, with each constraint annotated to its risk source.

**Measure** (analyze and monitor AI risks): The audit chain records every enforcement decision with the exact constraint violations, distances, solver methods, and routing decisions. The breaker's overload score provides continuous risk measurement. The trusted context provider injects live infrastructure telemetry into every enforcement cycle, enabling real-time risk assessment.

**Manage** (prioritize and act on AI risks): The enforcement engine is the Manage function — it acts on risk by approving, projecting, or rejecting every action. The breaker state machine provides automated escalation (CLOSED → THROTTLED → SAFE_STOP). The HITL layer provides human-authorized escalation for actions that exceed configured thresholds.

A NIST policy pack would map each constraint to the RMF function and sub-function it satisfies, providing a direct compliance crosswalk.

### EU AI Act

The EU AI Act establishes requirements for high-risk AI systems. The global default policy already references four articles:

**Article 9 (Risk management system):** The constraint geometry is the risk management system for the actuation layer. Each constraint encodes a risk control measure. The policy contract chain provides the "continuously iterated" risk management process the article requires — each policy version is a documented iteration with a cryptographic link to its predecessor.

**Article 9(4) (Residual risk measures):** The enforcement guarantee is the residual risk measure — it bounds the maximum impact of any AI action to the constraint geometry, regardless of model behavior. The breaker provides automated degradation when risk indicators exceed thresholds.

**Article 12 (Record-keeping):** The audit chain satisfies the record-keeping requirement. Every enforcement decision, every human review, every policy version, and every budget transaction is recorded with SHA-256 hash-chain integrity. The records are automatically generated, tamper-evident, and exportable.

**Article 14 (Human oversight):** The SupervisedGovernor satisfies the human oversight requirement. The three review profiles (ADVISORY, SUPERVISORY, MANDATORY) provide configurable levels of human involvement. The TOCTOU re-enforcement ensures that human-approved actions are still constraint-satisfying at execution time. The authenticated reviewer requirement ensures that oversight is attributable.

**Article 15 (Accuracy, robustness, cybersecurity):** The formal verification (two proof assistants, 3,732 structural checks) provides accuracy evidence. The fail-closed architecture provides robustness. The trusted context provider mitigates the cybersecurity risk of the AI manipulating its own evaluation context.

An EU AI Act policy pack would include constraints mapping to each applicable article, with the metadata field referencing the specific article and paragraph.

### ISO/IEC 42001 (AI Management System)

ISO/IEC 42001 requires organizations to establish, implement, maintain, and continually improve an AI management system. The standard's Annex A controls map to Numerail:

**A.2 (Policies for AI):** Policy packs are the policies. Versioned, digested, auditable.

**A.5 (AI system life cycle):** The policy contract chain records the constraint geometry at every stage of the AI system's lifecycle. Policy changes are chain-linked and tamper-evident.

**A.7 (Data for AI systems):** Trusted context injection ensures that the data used for enforcement decisions (telemetry, budgets, time) comes from authoritative sources, not from the AI system being governed.

**A.10 (AI system operation and monitoring):** The governor lifecycle, the breaker state machine, and the audit chain provide continuous operation monitoring with automated response to anomalies.

### Center for Humane Technology — The AI Roadmap (2026)

The Center for Humane Technology's report *The AI Roadmap: How We Ensure AI Serves Humanity* (2026) lays out seven principles for how AI should be built and governed, with actionable solutions across norms, laws, and product design. Three of its principles map directly to Numerail capabilities, and several of its specific design recommendations describe requirements that Numerail satisfies.

**Principle 1 (AI should be built safely and transparently)** calls for "high reliability and deterministic behavior," stating that AI systems should be engineered to operate reliably with clear safeguards that ensure failures are contained and predictable, like with other safety-critical technologies such as airplanes. Numerail's enforcement guarantee is deterministic and provably reliable. The fly-by-wire analogy the report invokes is the same design primitive Numerail implements. The report also calls for protected containers or restrictions for AI agents — standardized, contained environments that do not give agents full control over sensitive data. Numerail's constraint geometry is exactly this: a mathematically defined container for AI agent actuation. The report's call for mandatory predeployment testing and risk management maps to the policy contract system, which records each constraint specification as a versioned, digested artifact. Its call for independent audit and certification schemes maps to the audit chain and the policy contract chain, which provide cryptographic proof of compliance that any third party can verify.

**Principle 1 (Design recommendations)** calls for scrutable and plain language explainability — actions validated through traceable logs, reasoning steps interpreted by users, and decisions retaining auditable records. The Numerail audit chain satisfies this: every enforcement decision is recorded with the exact constraint violations, the solver method, the distance, the routing decision, and the human review outcome if applicable.

**Principle 5 (AI innovation should not come at the expense of our rights and freedom)** calls for limiting data collection and minimizing data leakage. The hosted enforcement API model addresses this: the action vector contains only numerical magnitudes, never the content of the action. The regulator enforces constraints on what the AI is allowed to do without ever seeing what the AI is saying or reading.

**Principle 6 (AI should have internationally agreed-upon limits)** calls for technical verification and developing coordination tactics and verification methods to ensure compliance. Numerail's policy contract system provides exactly this: a content-addressable, cryptographically linked policy chain that any jurisdiction can publish and any deployer can load, with independent digest verification proving which constraints were in force at any point in time. The report's analogy to nuclear nonproliferation verification is apt — the policy contract chain serves the same function as an inspection record, but for AI actuation constraints.

**Principle 7 (AI power should be balanced in society)** calls for ensuring that no single company or actor should control the trajectory of AI. Numerail is open source (MIT license), the enforcement engine is a single-file mathematical kernel with a formal proof, and any organization — regulator, company, or individual — can run it independently. The constraint geometry is auditable by anyone who can read the policy pack. The enforcement guarantee does not depend on trusting any vendor, platform, or AI company. It depends on the mathematics.

The CHT report provides a comprehensive framework for why systems like Numerail are needed. Numerail provides the technical implementation for several of the report's most specific recommendations. The combination — CHT's governance principles defining what society needs from AI, and Numerail's enforcement engine providing a provably correct mechanism to deliver it — represents a concrete path from principle to practice.

Source: Center for Humane Technology, "The AI Roadmap: How We Ensure AI Serves Humanity," 2026. Available at https://www.datocms-assets.com/160835/1774623817-cht_report_theairoadmap.pdf

### Domain-Specific Regulations

Any regulatory body with quantifiable requirements can express them as Numerail constraints:

**Financial services (SEC, FCA, ESMA, MAS):** Position limits, leverage caps, trading velocity limits, client exposure thresholds, concentration limits. All expressible as linear constraints on the trade vector. Joint risk limits (combined exposure across instruments) expressible as SOCP or quadratic constraints.

**Healthcare (FDA, EMA):** Dosage limits, interaction constraints, treatment frequency caps, resource allocation bounds for AI-assisted clinical decision support. Constraints can encode formulary limits, contra-indication rules (as coupling constraints that prevent certain combinations), and per-patient resource budgets.

**Energy (FERC, NERC, ENTSO-E):** Grid stability constraints, generation/load balance requirements, reserve margins, ramp rate limits for AI-controlled grid management. The PSD constraint can encode coupled stability requirements across multiple grid nodes.

**Autonomous vehicles (NHTSA, UNECE):** Velocity limits, acceleration bounds, following distance constraints, lane-change rate limits for AI driving systems. The constraint geometry is the operational design domain expressed mathematically.

---

## Implementation Path for Regulatory Bodies

### Phase 1: Publish a reference policy pack

The regulatory body translates its existing requirements into a Numerail policy pack. This requires: identifying the regulated dimensions (what numerical quantities the AI agent controls), defining the constraints (what combinations of values are permissible), declaring trusted fields (what values must come from the infrastructure rather than the AI), and specifying the enforcement mode and routing thresholds. The policy pack is published as a versioned document alongside the regulation it implements.

### Phase 2: Establish a compliance verification protocol

The regulatory body defines how deployers demonstrate compliance. The simplest protocol: the deployer provides the audit chain export and the policy contract chain from their Numerail instance. The regulator verifies that the policy contract chain includes the regulator's published policy pack (matching the digest), and that the audit chain shows every action was enforced against that policy with no APPROVE or PROJECT outputs that violated any constraint — which is guaranteed by the enforcement engine and verifiable from the audit records.

### Phase 3: Operate a hosted enforcement API (optional)

For jurisdictions or domains where centralized enforcement is preferable, the regulatory body deploys Numerail as a public API service. This provides real-time enforcement, central audit, and the ability to update constraints dynamically in response to changing conditions (market volatility, grid stress, public health emergencies).

### Phase 4: Certify third-party implementations

As the ecosystem matures, the regulatory body can certify third-party Numerail deployments — verifying that the deployment loads the correct policy pack, maintains audit chain integrity, and operates the HITL layer at the required review profile. The policy contract system's cryptographic provenance makes this certification auditable after the fact without requiring real-time access to the deployment.

---

## What This Means for Regulated Organizations

An organization deploying AI agents in a regulated domain can:

1. Load the regulator's published policy pack into their Numerail instance.
2. Enforce every AI action against the regulator's constraint geometry with a mathematical guarantee.
3. Export the audit chain and policy contract chain as compliance evidence.
4. Demonstrate to auditors that every action satisfied every regulatory constraint at the time of execution, with cryptographic proof of which constraints were in force.

The compliance question changes from "did the AI behave correctly?" (which is subjective and untestable in the limit) to "did every action satisfy the published constraints?" (which is objective, verifiable, and guaranteed by the enforcement engine).

This is the same shift that occurred in aviation when flight data recorders and fly-by-wire envelopes replaced pilot self-reporting as the basis for safety compliance. The regulator defines the envelope. The engine enforces it. The recorder proves it. The model of compliance is: trust the proof, not the pilot.
