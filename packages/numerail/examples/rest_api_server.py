# =========================================================================
# ⚠️  DEVELOPMENT ONLY
#
# This server has no authentication, no rate limiting, and no TLS.
# Do not expose to untrusted networks or run in production.
# For production use, implement authentication via NumerailRuntimeService
# with an AuthorizationService provider.
# =========================================================================

"""Numerail REST API server — wraps NumerailSystemLocal in a FastAPI application.

Three endpoints
---------------
  POST /enforce
      Body:    {"action": {field: value, ...}, "action_id": "...", "execution_topic": "..."}
      Header:  X-Trusted-Context: {"current_gpu_util": 0.55, "current_api_util": 0.40}
      Returns: {"action_id", "decision", "enforced_values", "audit_hash", "feedback"}

  POST /rollback/{action_id}
      Returns: {"rolled_back": true, "audit_hash": "..."}
      Errors:  404 if action_id unknown, 409 if already rolled back

  GET /budgets
      Returns: {"remaining": {...}, "initial": {...}}

Trusted context
---------------
Supply real server-measured telemetry in the X-Trusted-Context request header as a
JSON object.  The server injects it via NumerailSystemLocal's trusted_context parameter,
overwriting whatever the caller supplied for those fields in the action body.  Fields
not listed in the policy's trusted_fields are ignored.

Setup and run
-------------
    cd packages/numerail
    pip install -e .
    pip install fastapi uvicorn

    python examples/rest_api_server.py
      -- or --
    uvicorn examples.rest_api_server:app --reload

The server listens on http://127.0.0.1:8000 by default.
Interactive docs: http://127.0.0.1:8000/docs

Run the companion client to see all endpoints exercised:
    python examples/rest_api_client.py
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from numerail.local import NumerailSystemLocal
from numerail.errors import AuthorizationError

logger = logging.getLogger("numerail.server")


# ═══════════════════════════════════════════════════════════════════════════
#  POLICY CONFIGURATION
#
#  Eight-dimensional schema matching the ai_resource_governor example:
#  six agent-proposed workload fields and two server-trusted telemetry
#  fields.  All four constraint types are active.
# ═══════════════════════════════════════════════════════════════════════════

_FIELDS = [
    "prompt_k",            # prompt-token ceiling, thousands
    "completion_k",        # completion-token ceiling, thousands
    "tool_calls",          # total tool-call allowance
    "external_api_calls",  # paid / side-effecting external API calls
    "gpu_seconds",         # GPU lease for this step
    "parallel_workers",    # max concurrent subtasks
    "current_gpu_util",    # server-trusted instantaneous GPU utilisation [0,1]
    "current_api_util",    # server-trusted instantaneous API utilisation [0,1]
]
_N   = len(_FIELDS)
_IDX = {f: i for i, f in enumerate(_FIELDS)}

_MAXES = {
    "prompt_k": 64.0, "completion_k": 16.0, "tool_calls": 40.0,
    "external_api_calls": 20.0, "gpu_seconds": 120.0, "parallel_workers": 16.0,
    "current_gpu_util": 1.0, "current_api_util": 1.0,
}


def _row(coeffs: dict) -> list[float]:
    r = [0.0] * _N
    for f, v in coeffs.items():
        r[_IDX[f]] = float(v)
    return r


_A, _b, _names = [], [], []

# Box bounds
for _f in _FIELDS:
    _A.append(_row({_f: 1.0}));  _b.append(_MAXES[_f]);     _names.append(f"max_{_f}")
    _A.append(_row({_f: -1.0})); _b.append(0.0);            _names.append(f"min_{_f}")

# Structural: external_api_calls <= tool_calls
_A.append(_row({"external_api_calls": 1.0, "tool_calls": -1.0}))
_b.append(0.0); _names.append("external_le_tool_calls")

# GPU headroom: util + workload/240 <= 0.90
_A.append(_row({"current_gpu_util": 1.0, "gpu_seconds": 1.0 / 240}))
_b.append(0.90); _names.append("gpu_headroom")

# API headroom: util + calls/40 <= 0.90
_A.append(_row({"current_api_util": 1.0, "external_api_calls": 1.0 / 40}))
_b.append(0.90); _names.append("api_headroom")

# Budget target rows — tightened by BudgetTracker as the shift progresses
_A.append(_row({"gpu_seconds": 1.0}))
_b.append(600.0); _names.append("remaining_gpu_day")

_A.append(_row({"external_api_calls": 1.0}))
_b.append(80.0); _names.append("remaining_api_day")

# Quadratic: resource energy bound — no peak-everything-at-once
_Q_DIAG = [1/64**2, 1/16**2, 1/40**2, 1/20**2, 1/120**2, 1/16**2, 0.0, 0.0]

# SOCP: burst envelope
_M_SOCP = np.zeros((3, _N))
_M_SOCP[0, _IDX["current_gpu_util"]]   = 1.0
_M_SOCP[0, _IDX["gpu_seconds"]]        = 1.0 / 240
_M_SOCP[1, _IDX["current_api_util"]]   = 1.0
_M_SOCP[1, _IDX["external_api_calls"]] = 1.0 / 40
_M_SOCP[2, _IDX["parallel_workers"]]   = 1.0 / 16

# PSD: coupled headroom
_A0_PSD = np.eye(2).tolist()
_A_PSD  = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(_N)]
_A_PSD[_IDX["gpu_seconds"]]        = [[-1.0/240, 0.0],     [0.0, 0.0]]
_A_PSD[_IDX["external_api_calls"]] = [[0.0, 0.0],          [0.0, -1.0/40]]
_A_PSD[_IDX["parallel_workers"]]   = [[0.0, 0.35/16],      [0.35/16, 0.0]]
_A_PSD[_IDX["current_gpu_util"]]   = [[-1.0, 0.0],         [0.0, 0.0]]
_A_PSD[_IDX["current_api_util"]]   = [[0.0, 0.0],          [0.0, -1.0]]

CONFIG = {
    "policy_id": "rest_api_demo_v1",
    "schema": {"fields": _FIELDS},
    "polytope": {"A": [list(r) for r in _A], "b": _b, "names": _names},
    "quadratic_constraints": [{
        "Q": np.diag(_Q_DIAG).tolist(), "a": [0.0] * _N,
        "b": 2.25, "name": "resource_energy",
    }],
    "socp_constraints": [{
        "M": _M_SOCP.tolist(), "q": [0.0] * 3, "c": [0.0] * _N,
        "d": 1.15, "name": "burst_envelope",
    }],
    "psd_constraints": [{
        "A0": _A0_PSD, "A_list": _A_PSD, "name": "coupled_headroom_psd",
    }],
    "trusted_fields": ["current_gpu_util", "current_api_util"],
    "enforcement": {
        "mode": "project",
        "dimension_policies": {
            "current_gpu_util": "forbidden",
            "current_api_util": "forbidden",
            "parallel_workers": "project_with_flag",
        },
        "routing_thresholds": {
            "silent": 0.05, "flagged": 1.5, "confirmation": 4.0, "hard_reject": 8.0,
        },
    },
    "budgets": [
        {
            "name": "gpu_day",
            "constraint_name": "remaining_gpu_day",
            "weight": {"gpu_seconds": 1.0},
            "initial": 600.0,
            "consumption_mode": "nonnegative",
        },
        {
            "name": "api_day",
            "constraint_name": "remaining_api_day",
            "weight": {"external_api_calls": 1.0},
            "initial": 80.0,
            "consumption_mode": "nonnegative",
        },
    ],
}

# Budget initial values for reporting — read once at startup
_BUDGET_INITIAL = {b["name"]: b["initial"] for b in CONFIG["budgets"]}


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class EnforceRequest(BaseModel):
    # All action field values.  Include every schema field; missing fields
    # default to 0.0 in the enforcement engine.
    action: dict[str, float]
    # Optional caller-supplied action ID for idempotency and rollback lookup.
    # If omitted the server generates one automatically.
    action_id: Optional[str] = None
    # Outbox routing topic — the server publishes to this topic on approve/project.
    execution_topic: Optional[str] = None


class EnforceResponse(BaseModel):
    action_id: str
    decision: str                              # "approve" | "project" | "reject"
    enforced_values: Optional[dict[str, Any]]  # None when decision == "reject"
    audit_hash: str
    feedback: dict[str, Any]


class RollbackResponse(BaseModel):
    rolled_back: bool
    audit_hash: str


class BudgetsResponse(BaseModel):
    remaining: dict[str, float]
    initial: dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════
#  APPLICATION LIFESPAN
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the NumerailSystemLocal once at startup.  All requests share
    # this single instance and its in-memory budget/audit/ledger state.
    app.state.numerail = NumerailSystemLocal(CONFIG)
    logger.info("Numerail system initialised (policy_id=%s)", CONFIG["policy_id"])
    yield
    # Nothing to tear down for in-memory state.


app = FastAPI(
    title="Numerail Enforcement API",
    description=(
        "Wraps NumerailSystemLocal in a REST interface. "
        "Supply trusted telemetry via the X-Trusted-Context header."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: parse trusted context header
# ═══════════════════════════════════════════════════════════════════════════

def _parse_trusted_context(raw: Optional[str]) -> Optional[dict[str, float]]:
    """Parse the X-Trusted-Context header value.

    Expected format: JSON object mapping field names to numeric values.
    Example:  {"current_gpu_util": 0.55, "current_api_util": 0.40}

    Returns None if the header is absent.  Raises HTTPException 422 if the
    header is present but not a valid JSON object.
    """
    if raw is None:
        return None
    try:
        ctx = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"X-Trusted-Context is not valid JSON: {exc}",
        )
    if not isinstance(ctx, dict):
        raise HTTPException(
            status_code=422,
            detail="X-Trusted-Context must be a JSON object, not a scalar or array.",
        )
    return {k: float(v) for k, v in ctx.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/enforce", response_model=EnforceResponse, summary="Enforce an action")
async def enforce(
    body: EnforceRequest,
    request: Request,
    # FastAPI reads this header automatically; None if absent.
    x_trusted_context: Optional[str] = Header(default=None),
):
    """Enforce a proposed action against the active policy.

    - The action fields are compared against all constraints.
    - APPROVE: the proposal already satisfies every constraint.
    - PROJECT: the nearest feasible point is returned in enforced_values.
    - REJECT:  the proposal could not be made feasible (or was blocked by policy).

    Trusted context: supply server-authoritative telemetry values in the
    X-Trusted-Context header.  These override the corresponding fields in the
    action body before enforcement, preventing the caller from misrepresenting
    infrastructure state to inflate their own admissibility.
    """
    numerail: NumerailSystemLocal = request.app.state.numerail

    # Parse the optional trusted context header.
    trusted_ctx = _parse_trusted_context(x_trusted_context)

    try:
        result = numerail.enforce(
            body.action,
            action_id=body.action_id,
            trusted_context=trusted_ctx,
            execution_topic=body.execution_topic,
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during enforcement")
        raise HTTPException(status_code=500, detail=str(exc))

    return EnforceResponse(
        action_id=result["action_id"],
        decision=result["decision"],
        enforced_values=result["enforced_values"],
        audit_hash=result["audit_hash"],
        feedback=result["feedback"],
    )


@app.post(
    "/rollback/{action_id}",
    response_model=RollbackResponse,
    summary="Roll back a previously enforced action",
)
async def rollback(action_id: str, request: Request):
    """Roll back an action by its action_id.

    Restores the exact budget delta that was recorded at enforcement time
    (Theorem 6: rollback restoration).  Appends a rollback record to the
    audit chain.

    Returns 404 if the action_id is not found in the ledger.
    Returns 409 if the action has already been rolled back.
    """
    numerail: NumerailSystemLocal = request.app.state.numerail

    try:
        result = numerail.rollback(action_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"action_id {action_id!r} not found in ledger.",
        )
    except ValueError as exc:
        # "action_id already rolled back"
        raise HTTPException(status_code=409, detail=str(exc))
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return RollbackResponse(rolled_back=result.rolled_back, audit_hash=result.audit_hash)


@app.get("/budgets", response_model=BudgetsResponse, summary="Current budget status")
async def budgets(request: Request):
    """Return remaining and initial budget values for all configured budgets.

    Budget bounds tighten monotonically as the shift progresses (Theorem 5).
    Rollbacks restore the exact consumed delta for that action.
    """
    numerail: NumerailSystemLocal = request.app.state.numerail
    return BudgetsResponse(
        remaining=numerail.budget_remaining,
        initial=_BUDGET_INITIAL,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "examples.rest_api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
