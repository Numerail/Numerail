"""Numerail live demo — FastAPI backend.

Initialises a full Numerail + StateTransitionGovernor stack, runs the
scripted simulation in a background thread, and streams enforcement events
to the dashboard via WebSocket.

Run::

    pip install fastapi uvicorn websockets
    python backend.py

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from time import time_ns
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from numerail_ext.survivability.breaker import BreakerStateMachine
from numerail_ext.survivability.governor import StateTransitionGovernor
from numerail_ext.survivability.local_backend import LocalNumerailBackend
from numerail_ext.survivability.transition_model import IncidentCommanderTransitionModel
from numerail_ext.survivability.types import (
    BreakerThresholds,
    TelemetrySnapshot,
    WorkloadRequest,
)
from simulate import run_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------

_DEMO_DURATION_S = 120.0
_BOOTSTRAP_BUDGETS = {
    "gpu_shift":          3600.0,
    "external_api_shift":  500.0,
    "mutation_shift":      100.0,
}
_THRESHOLDS = BreakerThresholds(
    trip_score=0.50,
    reset_score=0.25,
    safe_stop_score=0.80,
)
_FRESHNESS_NS = 5_000_000_000  # 5 seconds

# ---------------------------------------------------------------------------
# Minimal in-process mock helpers (mirrors test_integration.py pattern)
# ---------------------------------------------------------------------------


class _MockReservationMgr:
    def acquire(self, *a, **kw): return True
    def commit(self, *a, **kw): pass
    def release(self, *a, **kw): pass


class _MockDigestor:
    def digest(self, *a, **kw) -> str:
        return hashlib.sha256(b"demo").hexdigest()


# ---------------------------------------------------------------------------
# Demo state (shared between simulation thread and WebSocket broadcast)
# ---------------------------------------------------------------------------

class DemoState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self.budget_remaining: Dict[str, float] = dict(_BOOTSTRAP_BUDGETS)
        self.breaker_mode: str = "CLOSED"
        self.overload_score: float = 0.0
        self.phase: str = "nominal"
        self.step: int = 0
        self.running: bool = False
        self.finished: bool = False
        self.policy_digest: Optional[str] = None
        self.policy_id: Optional[str] = None
        self.audit_records: List[Dict[str, Any]] = []

        # Rolling telemetry for sparklines (last 60 readings)
        self.telemetry_history: List[Dict[str, float]] = []

    def push_event(self, ev: Dict[str, Any]) -> None:
        with self.lock:
            self.events.append(ev)
            # Keep last 500 events in memory
            if len(self.events) > 500:
                self.events = self.events[-500:]

    def snapshot_for_ws(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "budget_remaining": dict(self.budget_remaining),
                "breaker_mode": self.breaker_mode,
                "overload_score": self.overload_score,
                "phase": self.phase,
                "step": self.step,
                "running": self.running,
                "finished": self.finished,
                "policy_digest": self.policy_digest,
                "policy_id": self.policy_id,
                "telemetry_history": list(self.telemetry_history[-60:]),
            }


_state = DemoState()

# ---------------------------------------------------------------------------
# Governor setup
# ---------------------------------------------------------------------------

def _build_governor() -> StateTransitionGovernor:
    return StateTransitionGovernor(
        backend=LocalNumerailBackend(),
        transition_model=IncidentCommanderTransitionModel(freshness_ns=_FRESHNESS_NS),
        reservation_mgr=_MockReservationMgr(),
        digestor=_MockDigestor(),
        thresholds=_THRESHOLDS,
        bootstrap_budgets=dict(_BOOTSTRAP_BUDGETS),
    )


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------

def _run_demo_loop(governor: StateTransitionGovernor) -> None:
    """Background thread: runs the scripted simulation and writes to _state."""
    global _state

    _state.running = True
    counter = 0

    # Capture initial policy info
    try:
        raw = governor.backend._local.audit_records  # type: ignore[attr-defined]
        _state.policy_id = "incident-commander-v1"
    except Exception:
        pass

    try:
        for request, snapshot, phase in run_simulation(duration_seconds=_DEMO_DURATION_S):
            counter += 1
            action_id = f"demo_{counter:04d}"

            try:
                step = governor.enforce_next_step(
                    request=request,
                    snapshot=snapshot,
                    action_id=action_id,
                    execution_topic="demo.enforcement",
                )
            except Exception as exc:
                logger.warning("enforce_next_step raised: %s", exc)
                continue

            # Extract results
            breaker = step.breaker
            nr = step.numerail_result

            decision    = nr.get("decision", "REJECT")
            enforced    = nr.get("enforced_values", {})
            feedback    = nr.get("feedback", [])
            audit_hash  = nr.get("audit_hash", "")

            # Compute budget_remaining from governor backend
            try:
                budget = governor.backend.budget_remaining()  # type: ignore[attr-defined]
            except Exception:
                budget = dict(_state.budget_remaining)

            # Telemetry snapshot for history
            telem_point = {
                "t": time.time(),
                "gpu":   snapshot.current_gpu_util,
                "api":   snapshot.current_api_util,
                "db":    snapshot.current_db_util,
                "queue": snapshot.current_queue_util,
                "err":   snapshot.current_error_rate_pct,
                "score": breaker.overload_score,
            }

            # Audit record (append a simplified version)
            audit_entry = {
                "step": counter,
                "action_id": action_id,
                "decision": decision,
                "audit_hash": audit_hash,
                "breaker": breaker.mode.value if hasattr(breaker.mode, "value") else str(breaker.mode),
            }

            with _state.lock:
                _state.phase          = phase
                _state.step           = counter
                _state.breaker_mode   = breaker.mode.value if hasattr(breaker.mode, "value") else str(breaker.mode)
                _state.overload_score = round(breaker.overload_score, 4)
                _state.budget_remaining = budget
                _state.telemetry_history.append(telem_point)
                if len(_state.telemetry_history) > 120:
                    _state.telemetry_history = _state.telemetry_history[-120:]
                _state.audit_records.append(audit_entry)
                if len(_state.audit_records) > 200:
                    _state.audit_records = _state.audit_records[-200:]

            # Build broadcast event
            ev = {
                "type": "enforcement",
                "step": counter,
                "phase": phase,
                "action_id": action_id,
                "decision": decision,
                "breaker_mode": _state.breaker_mode,
                "overload_score": _state.overload_score,
                "enforced": {k: round(float(v), 4) for k, v in enforced.items()} if enforced else {},
                "proposed": {k: round(float(v), 4) for k, v in request.as_action_dict().items()},
                "feedback": list(feedback),
                "audit_hash": audit_hash[:16] + "…" if audit_hash else "",
                "budget_remaining": {k: round(float(v), 2) for k, v in budget.items()},
                "telemetry": {
                    "gpu":   snapshot.current_gpu_util,
                    "api":   snapshot.current_api_util,
                    "db":    snapshot.current_db_util,
                    "queue": snapshot.current_queue_util,
                    "err":   snapshot.current_error_rate_pct,
                },
                "ts": time.time(),
            }
            _state.push_event(ev)
            _notify_ws_listeners(ev)

    finally:
        with _state.lock:
            _state.running  = False
            _state.finished = True
        _notify_ws_listeners({"type": "finished"})
        logger.info("Demo simulation finished after %d steps.", counter)


# ---------------------------------------------------------------------------
# WebSocket broadcast helpers
# ---------------------------------------------------------------------------

_ws_clients: List[asyncio.Queue] = []
_ws_lock = threading.Lock()


def _notify_ws_listeners(ev: Dict[str, Any]) -> None:
    payload = json.dumps(ev)
    with _ws_lock:
        for q in list(_ws_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Numerail Live Demo", docs_url=None, redoc_url=None)

_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_DASHBOARD_PATH, media_type="text/html")


@app.get("/audit")
async def audit() -> JSONResponse:
    with _state.lock:
        records = list(_state.audit_records[-100:])
    return JSONResponse({"records": records})


@app.get("/policy")
async def policy() -> JSONResponse:
    info = {
        "policy_id":      _state.policy_id,
        "policy_digest":  _state.policy_digest,
        "bootstrap_budgets": _BOOTSTRAP_BUDGETS,
        "thresholds": {
            "trip_score":      _THRESHOLDS.trip_score,
            "reset_score":     _THRESHOLDS.reset_score,
            "safe_stop_score": _THRESHOLDS.safe_stop_score,
        },
        "duration_seconds": _DEMO_DURATION_S,
    }
    return JSONResponse(info)


@app.get("/state")
async def state() -> JSONResponse:
    return JSONResponse(_state.snapshot_for_ws())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _ws_lock:
        _ws_clients.append(q)

    # Send current state immediately on connect
    await ws.send_text(json.dumps({"type": "state", **_state.snapshot_for_ws()}))

    # Send recent event history so the feed populates instantly
    with _state.lock:
        recent = list(_state.events[-30:])
    for ev in recent:
        await ws.send_text(json.dumps(ev))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
                await ws.send_text(msg)
            except asyncio.TimeoutError:
                # Keepalive ping
                await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        with _ws_lock:
            try:
                _ws_clients.remove(q)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Startup: launch simulation thread
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    governor = _build_governor()

    # Try to capture policy digest
    try:
        from numerail_ext.survivability.contract import NumerailPolicyContract  # type: ignore
        # digest is available after the first enforce call in real usage;
        # for display purposes use the digestor
        _state.policy_digest = hashlib.sha256(b"incident-commander-v1-demo").hexdigest()[:16] + "…"
    except Exception:
        pass

    t = threading.Thread(target=_run_demo_loop, args=(governor,), daemon=True)
    t.start()
    logger.info("Demo simulation thread started.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
