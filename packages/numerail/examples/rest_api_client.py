"""Numerail REST API client — exercises all three server endpoints.

Start the server first
-----------------------
    cd packages/numerail
    pip install -e .
    pip install fastapi uvicorn

    python examples/rest_api_server.py
      -- or --
    uvicorn examples.rest_api_server:app --reload

Then in a second terminal run this client:
    cd packages/numerail
    python examples/rest_api_client.py

No extra dependencies beyond the Python standard library (urllib + json).

What this script demonstrates
-------------------------------
  1. GET /budgets  — initial budget snapshot
  2. POST /enforce — a modest proposal that is approved as-is
  3. POST /enforce — a heavy proposal that gets projected (values reduced)
  4. POST /enforce — trusted context injected via X-Trusted-Context header;
                     server overrides the agent-supplied utilisation values
  5. GET /budgets  — budget after three approved/projected actions
  6. POST /rollback/{action_id} — restore the first action's budget delta
  7. GET /budgets  — budget after rollback (first delta restored)
  8. POST /enforce — a proposal that is rejected outright (infeasible region)
  9. POST /rollback/{action_id} — 404 for an unknown action_id
 10. POST /rollback/{action_id} — 409 on a double-rollback attempt
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "http://127.0.0.1:8000"


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _request(
    method: str,
    path: str,
    body: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send an HTTP request; return (status_code, parsed_json)."""
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _sep(label: str = "") -> None:
    line = "-" * 72
    if label:
        pad = (72 - len(label) - 2) // 2
        line = "-" * pad + " " + label + " " + "-" * (72 - pad - len(label) - 2)
    print(line)


def _print_response(status: int, body: dict) -> None:
    print(f"  HTTP {status}")
    print("  " + json.dumps(body, indent=2).replace("\n", "\n  "))


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("=" * 72)
    print("  Numerail REST API Client Demo")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Initial budget snapshot
    # ------------------------------------------------------------------
    _sep("1. GET /budgets (initial)")
    status, body = _request("GET", "/budgets")
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"

    # ------------------------------------------------------------------
    # 2. POST /enforce — modest proposal, expect APPROVE
    # ------------------------------------------------------------------
    _sep("2. POST /enforce  -- modest proposal (expect APPROVE)")
    action_modest = {
        "prompt_k": 8.0,
        "completion_k": 2.0,
        "tool_calls": 5.0,
        "external_api_calls": 3.0,
        "gpu_seconds": 30.0,
        "parallel_workers": 2.0,
        # These utilisation values are agent-supplied; no trusted context header
        # is sent here, so they pass through as-is.
        "current_gpu_util": 0.30,
        "current_api_util": 0.20,
    }
    status, body = _request(
        "POST",
        "/enforce",
        body={"action": action_modest, "action_id": "step-1"},
    )
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"
    action_id_1 = body["action_id"]
    print(f"  decision: {body['decision']}   action_id: {action_id_1}")

    # ------------------------------------------------------------------
    # 3. POST /enforce — heavy proposal, expect PROJECT
    # ------------------------------------------------------------------
    _sep("3. POST /enforce  -- heavy proposal (expect PROJECT)")
    action_heavy = {
        "prompt_k": 60.0,       # near the 64 k ceiling
        "completion_k": 15.0,   # near the 16 k ceiling
        "tool_calls": 38.0,
        "external_api_calls": 18.0,
        "gpu_seconds": 110.0,
        "parallel_workers": 15.0,
        "current_gpu_util": 0.50,
        "current_api_util": 0.45,
    }
    status, body = _request(
        "POST",
        "/enforce",
        body={"action": action_heavy, "action_id": "step-2"},
    )
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  decision: {body['decision']}   action_id: {body['action_id']}")
    if body["enforced_values"]:
        ev = body["enforced_values"]
        print(
            f"  enforced gpu_seconds={ev.get('gpu_seconds', '?'):.1f}  "
            f"parallel_workers={ev.get('parallel_workers', '?'):.1f}"
        )

    # ------------------------------------------------------------------
    # 4. POST /enforce — server injects trusted context via header
    # ------------------------------------------------------------------
    _sep("4. POST /enforce  -- trusted context header (server overrides util)")
    #
    # The agent claims low utilisation (0.10/0.10) to inflate admissibility.
    # The server-side header supplies the real measurements (0.75/0.65).
    # The kernel uses the header values, not the agent-supplied ones.
    # With gpu_util=0.75 and gpu_seconds=90, the gpu_headroom constraint
    # (util + seconds/240 <= 0.90) forces gpu_seconds down significantly.
    #
    action_sneaky = {
        "prompt_k": 10.0,
        "completion_k": 4.0,
        "tool_calls": 8.0,
        "external_api_calls": 5.0,
        "gpu_seconds": 90.0,
        "parallel_workers": 4.0,
        # Agent claims low utilisation — server will override these.
        "current_gpu_util": 0.10,
        "current_api_util": 0.10,
    }
    trusted_header = json.dumps({"current_gpu_util": 0.75, "current_api_util": 0.65})
    status, body = _request(
        "POST",
        "/enforce",
        body={"action": action_sneaky, "action_id": "step-3"},
        headers={"X-Trusted-Context": trusted_header},
    )
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  decision: {body['decision']}   action_id: {body['action_id']}")
    if body["enforced_values"]:
        ev = body["enforced_values"]
        print(
            f"  enforced current_gpu_util={ev.get('current_gpu_util', '?'):.2f}  "
            f"(agent said 0.10, server said 0.75)"
        )
        print(
            f"  enforced gpu_seconds={ev.get('gpu_seconds', '?'):.1f}  "
            f"(reduced from 90 by gpu_headroom constraint)"
        )

    # ------------------------------------------------------------------
    # 5. Budget after three actions
    # ------------------------------------------------------------------
    _sep("5. GET /budgets  -- after three enforced actions")
    status, body = _request("GET", "/budgets")
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"

    # ------------------------------------------------------------------
    # 6. Rollback the first action
    # ------------------------------------------------------------------
    _sep(f"6. POST /rollback/{action_id_1}  -- restore first action's budget delta")
    status, body = _request("POST", f"/rollback/{action_id_1}")
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  rolled_back: {body['rolled_back']}   audit_hash: {body['audit_hash'][:16]}...")

    # ------------------------------------------------------------------
    # 7. Budget after rollback
    # ------------------------------------------------------------------
    _sep("7. GET /budgets  -- after rollback (step-1 delta restored)")
    status, body = _request("GET", "/budgets")
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"

    # ------------------------------------------------------------------
    # 8. POST /enforce — infeasible: gpu_util=0.95 makes gpu_seconds region empty
    # ------------------------------------------------------------------
    _sep("8. POST /enforce  -- infeasible region (expect REJECT)")
    #
    # gpu_headroom: gpu_util + gpu_seconds/240 <= 0.90
    # With gpu_util=0.95 (injected by trusted context), any gpu_seconds > 0
    # violates the headroom constraint.  The feasible set is empty => REJECT.
    #
    action_infeasible = {
        "prompt_k": 5.0,
        "completion_k": 2.0,
        "tool_calls": 3.0,
        "external_api_calls": 1.0,
        "gpu_seconds": 60.0,
        "parallel_workers": 2.0,
        "current_gpu_util": 0.10,   # agent claim — server overrides
        "current_api_util": 0.20,
    }
    status, body = _request(
        "POST",
        "/enforce",
        body={"action": action_infeasible, "action_id": "step-reject"},
        headers={"X-Trusted-Context": json.dumps({"current_gpu_util": 0.95})},
    )
    _print_response(status, body)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  decision: {body['decision']}  (infeasible region — correct)")

    # ------------------------------------------------------------------
    # 9. Rollback an unknown action_id — expect 404
    # ------------------------------------------------------------------
    _sep("9. POST /rollback/nonexistent-id  -- expect 404")
    status, body = _request("POST", "/rollback/nonexistent-action-id")
    _print_response(status, body)
    assert status == 404, f"Expected 404, got {status}"
    print("  404 received as expected")

    # ------------------------------------------------------------------
    # 10. Double-rollback — expect 409
    # ------------------------------------------------------------------
    _sep(f"10. POST /rollback/{action_id_1}  -- double rollback (expect 409)")
    status, body = _request("POST", f"/rollback/{action_id_1}")
    _print_response(status, body)
    assert status == 409, f"Expected 409, got {status}"
    print("  409 received as expected")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _sep()
    print()
    print("All assertions passed.  All three endpoints exercised successfully.")
    print()


if __name__ == "__main__":
    main()
