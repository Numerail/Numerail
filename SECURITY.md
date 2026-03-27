# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in Numerail, please report it responsibly by emailing **trynumerail@gmail.com**. Do not open a public issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide an initial assessment within 7 days.

## Threat Model

Numerail's enforcement guarantee covers the mathematical properties of the `enforce()` function — specifically, that every APPROVE or PROJECT output satisfies every active constraint to within tolerance τ.

### In Scope

The following are considered security-relevant and should be reported:

- **Guarantee bypass:** An `enforce()` output with result APPROVE or PROJECT where the enforced vector violates an active constraint beyond tolerance τ.
- **HITL authentication bypass:** A `SupervisedGovernor` accepting a `HumanDecision` where `authenticated` is False.
- **Audit chain corruption:** A modification, insertion, or deletion of an audit record that is not detected by `AuditChain.verify()`.
- **Trusted context bypass:** The AI influencing a trusted field value that should be overwritten by the `TrustedContextProvider`.
- **Policy contract digest collision:** Two distinct policy configurations producing the same SHA-256 digest.

### Out of Scope

The following are specification or deployment issues, not vulnerabilities in the enforcement engine:

- **Specification errors:** Constraint values that do not match the deployer's intent (e.g., a cap set to 5,000 when 50 was intended).
- **Telemetry inaccuracy:** The `TrustedContextProvider` correctly injecting values from a telemetry source that reports incorrect data.
- **Deployment configuration:** Exposed ports, missing authentication on custom API wrappers, or misconfigured firewall rules.
- **Model behavior:** The AI proposing adversarial or malicious actions — the enforcement engine is designed to handle arbitrary inputs, including adversarial ones.

## Known Limitations

The following are documented architectural limitations, not vulnerabilities:

- **Audit chain genesis hash inconsistency:** `AuditChain` (engine.py) uses `""` (empty string) as the genesis `prev_hash`, while `_HitlAuditChain` (hitl.py) uses `"0" * 64`. Both are valid sentinels. Verification tools should handle both.
- **Policy contract signature field:** The `signature` field on `NumerailPolicyContract` is a placeholder — no signing implementation exists. The `digest` field (SHA-256) is the sole integrity verification mechanism.
- **Non-monotonic default time source:** `DefaultTimeProvider` uses `time.time_ns() // 1_000_000` (wall clock), which can go backward on NTP adjustments. Production deployments should implement a `TrustedContextProvider` using a monotonic clock source.
- **No multi-tenant authorization in local mode:** `NumerailSystemLocal.rollback()` accepts any `action_id` without verifying caller authorization. The production `NumerailRuntimeService` layer provides authorization via the `AuthorizationService` protocol.
- **REST API example has no authentication:** The `rest_api_server.py` example is for development only and should not be exposed to untrusted networks.
- **Constraint geometry is reconstructible:** An attacker with access to the enforcement API can probe the feasible region by submitting many vectors and observing the results. Rate limiting and authentication mitigate this in production.

## Supported Versions

| Version | Supported |
|---|---|
| 5.0.x | ✅ |
| < 5.0 | ❌ |
