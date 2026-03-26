# numerail-ext

**Survivability extension for Numerail — supervisory degradation for AI actuation safety.**

Wraps the [Numerail](../numerail/README.md) enforcement kernel with a supervisory degradation layer that throttles, restricts, or halts AI operations based on real-time infrastructure telemetry.

## Install

```bash
pip install -e ../numerail    # core dependency
pip install -e .              # this package
```

## What it provides

**Breaker state machine** — Five operational modes with hysteretic transitions. Authority monotonically decreases through the mode hierarchy: CLOSED (full authority) → THROTTLED → HALF_OPEN → SAFE_STOP (latched) → OPEN (all authority suspended).

**Transition model** — Synthesizes conservative one-step envelopes from live telemetry, breaker mode, and remaining budgets. Caps are monotone non-increasing: tighter modes always mean tighter constraints.

**Policy builder** — Compiles envelopes into complete V5-compatible constraint configs (30 fields, ~80 linear constraints, quadratic, SOCP, and PSD constraints).

**Global default policy pack** — Locked-down-by-default policy aligned with EU AI Act Articles 9, 12, 14, and 15. The default is denial — every permission must be explicitly granted through constraint geometry.

**Governor** — 12-step enforce/commit lifecycle: evaluate telemetry → choose mode → synthesize envelope → compile policy → enforce through V5 → issue grant → validate receipt → verify next-state safety.

**Policy contract** — Content-addressable, chain-linked, tamper-detected policy interchange. SHA-256 digests, append-only version chains, portable verification using only Python's stdlib.

## Quickstart

```python
from numerail_ext import (
    BreakerStateMachine, BreakerThresholds, BreakerMode,
    IncidentCommanderTransitionModel,
    StateTransitionGovernor, LocalNumerailBackend,
    build_global_default, NumerailPolicyContract,
)

# Build a default policy and wrap it in a verifiable contract
config = build_global_default()
contract = NumerailPolicyContract.from_v5_config(
    config,
    author_id="governance-council",
    policy_id="global-default::v1.0",
)
assert contract.verify_digest()
```

## Tests

```bash
pytest tests/ -v                      # 217 tests (87 breaker + 120 contract + 10 integration)
pytest tests/test_integration.py -v  # integration only — full governor lifecycle
```

## Requires

- `numerail >= 5.0.0`
- Python ≥ 3.10
- numpy ≥ 1.21, scipy ≥ 1.7

## License

MIT — see [LICENSE](LICENSE).
