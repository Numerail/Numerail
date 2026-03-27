# Numerail Live Demo

A self-contained proof-of-concept that runs the full Numerail + `StateTransitionGovernor` stack on localhost with a real-time dashboard. No LLM API key is required — the "agent" is a deterministic Python simulation that walks through a scripted 5-phase narrative arc.

## What it shows

| Phase | Duration (at 120 s default) | What happens |
|---|---|---|
| **nominal** | 24 s | Low, stable load — breaker stays CLOSED, most proposals APPROVE |
| **escalation** | 24 s | Rising load — overload score climbs toward the trip threshold (0.50) |
| **spike** | 18 s | Sharp overload — breaker trips to THROTTLED or OPEN, proposals get PROJECTed or REJECTed |
| **recovery** | 24 s | Load shedding — score falls, breaker transitions through HALF_OPEN back to CLOSED |
| **steady** | 30 s | Post-recovery normal — enforcer back in CLOSED mode |

Live dashboard panels:

- **Header** — breaker mode badge (colour-coded: green/yellow/orange/red), current phase, WebSocket connection status
- **Budget bars** — GPU seconds, API calls, mutations remaining (shrink over time)
- **Telemetry gauges** — GPU, API, DB, queue utilisation; error rate
- **Overload score** — large numeric + bar, with trip (0.50) and safe-stop (0.80) threshold markers
- **2D feasible region canvas** — proposed (hollow) vs enforced (filled) points in GPU-seconds × external-API-calls space; PROJECT arrows show corrections
- **Enforcement feed** — scrolling list of every step with decision, phase, and audit hash prefix
- **Audit chain** — right panel listing every action ID, breaker mode, and full audit hash
- **Summary overlay** — appears at end with APPROVE/PROJECT/REJECT totals

## Setup

```bash
# Install demo dependencies (fastapi + uvicorn)
pip install fastapi uvicorn websockets

# The Numerail packages must already be installed:
#   pip install -e packages/numerail
#   pip install -e packages/numerail_ext
```

## Run

```bash
cd packages/numerail/examples/live_demo
python backend.py
```

Then open **http://localhost:8000** in a browser. The simulation starts automatically and runs for 120 seconds.

## REST endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the dashboard HTML |
| `GET /state` | Current snapshot: breaker mode, phase, budgets, overload score |
| `GET /audit` | Last 100 audit chain records |
| `GET /policy` | Policy configuration: budgets, thresholds, duration |
| `WS  /ws` | Live enforcement event stream |

## Architecture

```
simulate.py          ← scripted 5-phase generator (WorkloadRequest + TelemetrySnapshot)
    │
    ▼
backend.py           ← FastAPI + background thread
    │  StateTransitionGovernor (full production stack)
    │      LocalNumerailBackend → NumerailSystemLocal → engine.py
    │      IncidentCommanderTransitionModel
    │      BreakerStateMachine (thresholds: trip=0.50, reset=0.25, safe_stop=0.80)
    │
    ├── GET /          → dashboard.html
    ├── GET /state     → current system state
    ├── GET /audit     → audit chain records
    ├── GET /policy    → policy info
    └── WS  /ws        → enforcement event stream
             │
             ▼
    dashboard.html    ← self-contained SPA (no external CDN, dark theme)
```

## Configuration

All demo parameters are constants at the top of `backend.py`:

```python
_DEMO_DURATION_S    = 120.0
_BOOTSTRAP_BUDGETS  = { "gpu_shift": 3600.0, "external_api_shift": 500.0, "mutation_shift": 100.0 }
_THRESHOLDS         = BreakerThresholds(trip_score=0.50, reset_score=0.25, safe_stop_score=0.80)
_FRESHNESS_NS       = 5_000_000_000   # 5 seconds
```

To run a longer demo, change `_DEMO_DURATION_S`. To make the breaker trip more aggressively, lower `trip_score`.

## Notes

- The simulation is deterministic and reproducible — the same 5-phase arc runs every time.
- All enforcement happens in-process with in-memory state (no database, no network calls).
- The dashboard reconnects automatically if the WebSocket drops.
- The summary overlay appears when the simulation finishes; click **Restart Demo** to reload.
