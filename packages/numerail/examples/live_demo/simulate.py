"""Scripted 5-phase simulation for the Numerail live demo.

No LLM API key required — the "agent" is a deterministic Python simulation
that walks through a narrative arc: nominal → escalation → spike → recovery → steady.

Usage::

    from simulate import run_simulation
    for request, snapshot, phase in run_simulation(duration_seconds=120.0):
        ...

Each iteration yields:
    request   : WorkloadRequest   — AI-proposed values
    snapshot  : TelemetrySnapshot — server-authoritative telemetry
    phase     : str               — human-readable phase name
"""

from __future__ import annotations

import math
import time
from time import time_ns
from typing import Generator, Tuple

from numerail_ext.survivability.types import TelemetrySnapshot, WorkloadRequest

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

# Each phase is (name, fraction_of_total_duration, profile_fn)
# profile_fn(t_norm) → (gpu_util, api_util, db_util, queue_util, error_rate_pct)
# where t_norm ∈ [0, 1] is fractional progress through the phase.

_PHASES = [
    ("nominal",     0.20),  # 24 s at 120 s default
    ("escalation",  0.20),  # 24 s
    ("spike",       0.15),  # 18 s
    ("recovery",    0.20),  # 24 s
    ("steady",      0.25),  # 30 s
]

_STEP_INTERVAL_S = 2.0  # seconds between simulation steps


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Phase telemetry profiles
# ---------------------------------------------------------------------------

def _nominal(t: float) -> Tuple[float, float, float, float, float]:
    """Low, stable utilisation — breaker stays CLOSED."""
    gpu  = _clamp(_lerp(0.25, 0.35, t) + 0.03 * math.sin(t * 6.28))
    api  = _clamp(_lerp(0.20, 0.30, t))
    db   = _clamp(0.30 + 0.05 * math.sin(t * 3.14))
    q    = _clamp(0.15 + 0.05 * t)
    err  = _clamp(_lerp(0.3, 0.6, t), 0.0, 100.0)
    return gpu, api, db, q, err


def _escalation(t: float) -> Tuple[float, float, float, float, float]:
    """Steadily rising load — overload score approaches trip threshold."""
    gpu  = _clamp(_lerp(0.38, 0.62, t))
    api  = _clamp(_lerp(0.32, 0.58, t))
    db   = _clamp(_lerp(0.38, 0.55, t))
    q    = _clamp(_lerp(0.22, 0.50, t))
    err  = _clamp(_lerp(0.6, 4.0, t), 0.0, 100.0)
    return gpu, api, db, q, err


def _spike(t: float) -> Tuple[float, float, float, float, float]:
    """Sharp overload spike — breaker trips to THROTTLED / OPEN."""
    gpu  = _clamp(_lerp(0.65, 0.95, t))
    api  = _clamp(_lerp(0.62, 0.92, t))
    db   = _clamp(_lerp(0.58, 0.82, t))
    q    = _clamp(_lerp(0.52, 0.85, t))
    err  = _clamp(_lerp(5.0, 18.0, t), 0.0, 100.0)
    return gpu, api, db, q, err


def _recovery(t: float) -> Tuple[float, float, float, float, float]:
    """Load shedding — utilisation drops back toward normal."""
    gpu  = _clamp(_lerp(0.90, 0.40, t))
    api  = _clamp(_lerp(0.88, 0.35, t))
    db   = _clamp(_lerp(0.78, 0.38, t))
    q    = _clamp(_lerp(0.80, 0.25, t))
    err  = _clamp(_lerp(15.0, 1.0, t), 0.0, 100.0)
    return gpu, api, db, q, err


def _steady(t: float) -> Tuple[float, float, float, float, float]:
    """Post-recovery steady state — breaker back to CLOSED."""
    gpu  = _clamp(0.30 + 0.06 * math.sin(t * 9.42))
    api  = _clamp(0.28 + 0.04 * math.sin(t * 6.28 + 1.0))
    db   = _clamp(0.32 + 0.05 * math.sin(t * 4.71))
    q    = _clamp(0.18 + 0.04 * t)
    err  = _clamp(0.4 + 0.3 * math.sin(t * 12.56), 0.0, 100.0)
    return gpu, api, db, q, err


_PROFILE_FNS = {
    "nominal":    _nominal,
    "escalation": _escalation,
    "spike":      _spike,
    "recovery":   _recovery,
    "steady":     _steady,
}


# ---------------------------------------------------------------------------
# Workload request profiles — what the "AI agent" proposes each phase
# ---------------------------------------------------------------------------

def _workload_for_phase(phase: str, t: float) -> WorkloadRequest:
    """Return a WorkloadRequest scaled to the current phase and progress."""

    if phase == "nominal":
        prompt_k   = _lerp(4.0,  6.0, t)
        completion_k = _lerp(1.5, 2.5, t)
        ext        = _clamp(_lerp(2.0, 4.0, t), 0.0, 20.0)
        cloud_mut  = _clamp(_lerp(0.5, 1.5, t), 0.0, ext)
        gpu_s      = _lerp(8.0,  12.0, t)
        parallel   = max(1.0, round(_lerp(1.0, 2.0, t)))
        pager      = 0.0

    elif phase == "escalation":
        prompt_k   = _lerp(6.0,  12.0, t)
        completion_k = _lerp(2.5, 5.0, t)
        ext        = _clamp(_lerp(4.0, 10.0, t), 0.0, 20.0)
        cloud_mut  = _clamp(_lerp(1.5, 4.0, t), 0.0, ext)
        gpu_s      = _lerp(12.0, 35.0, t)
        parallel   = max(1.0, round(_lerp(2.0, 5.0, t)))
        pager      = 0.0

    elif phase == "spike":
        prompt_k   = _lerp(14.0, 20.0, t)
        completion_k = _lerp(6.0, 10.0, t)
        ext        = _clamp(_lerp(10.0, 18.0, t), 0.0, 20.0)
        cloud_mut  = _clamp(_lerp(4.0, 8.0, t), 0.0, ext)
        gpu_s      = _lerp(40.0, 80.0, t)
        parallel   = max(1.0, round(_lerp(6.0, 12.0, t)))
        pager      = min(1.0, ext)  # start paging under load

    elif phase == "recovery":
        prompt_k   = _lerp(18.0, 6.0, t)
        completion_k = _lerp(8.0, 2.5, t)
        ext        = _clamp(_lerp(16.0, 3.0, t), 0.0, 20.0)
        cloud_mut  = _clamp(_lerp(6.0, 1.0, t), 0.0, ext)
        gpu_s      = _lerp(70.0, 12.0, t)
        parallel   = max(1.0, round(_lerp(10.0, 2.0, t)))
        pager      = 0.0

    else:  # steady
        prompt_k   = _lerp(5.0, 7.0, t)
        completion_k = _lerp(2.0, 3.0, t)
        ext        = _clamp(_lerp(2.0, 5.0, t), 0.0, 20.0)
        cloud_mut  = _clamp(_lerp(0.5, 2.0, t), 0.0, ext)
        gpu_s      = _lerp(10.0, 14.0, t)
        parallel   = max(1.0, round(_lerp(1.0, 3.0, t)))
        pager      = 0.0

    # Clamp comms to leave room: pager + comms <= ext
    comms = _clamp(0.0, 0.0, max(0.0, ext - pager))
    # internal_tool_calls must be >= external_api_calls
    internal = max(float(round(ext + 1.0)), ext)

    return WorkloadRequest(
        prompt_k=round(prompt_k, 2),
        completion_k=round(completion_k, 2),
        internal_tool_calls=round(internal, 1),
        external_api_calls=round(ext, 1),
        cloud_mutation_calls=round(cloud_mut, 1),
        gpu_seconds=round(gpu_s, 2),
        parallel_workers=float(int(parallel)),
        traffic_shift_pct=0.0,
        worker_scale_up_pct=0.0,
        feature_flag_changes=0.0,
        rollback_batch_pct=0.0,
        pager_notifications=round(pager, 1),
        customer_comms_count=round(comms, 1),
    )


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

_STATE_VERSION = 1


def _build_snapshot(gpu: float, api: float, db: float, q: float,
                    err: float) -> TelemetrySnapshot:
    """Build a TelemetrySnapshot from normalised utilisation values."""
    # Control-plane reserves shrink as load rises — less headroom under load
    load_factor = (gpu + api) / 2.0  # 0–1
    gpu_reserve = _clamp(60.0 * (1.0 - load_factor * 0.85), 5.0, 60.0)
    api_reserve = _clamp(20.0 * (1.0 - load_factor * 0.80), 2.0, 20.0)
    par_reserve = _clamp(8.0  * (1.0 - load_factor * 0.75), 1.0, 8.0)
    mut_reserve = _clamp(5.0  * (1.0 - load_factor * 0.80), 0.5, 5.0)

    return TelemetrySnapshot(
        state_version=_STATE_VERSION,
        observed_at_ns=time_ns(),
        current_gpu_util=round(gpu, 4),
        current_api_util=round(api, 4),
        current_db_util=round(db, 4),
        current_queue_util=round(q, 4),
        current_error_rate_pct=round(err, 4),
        ctrl_gpu_reserve_seconds=round(gpu_reserve, 2),
        ctrl_api_reserve_calls=round(api_reserve, 2),
        ctrl_parallel_reserve=round(par_reserve, 2),
        ctrl_cloud_mutation_reserve=round(mut_reserve, 2),
        gpu_disturbance_margin_seconds=15.0,
        api_disturbance_margin_calls=3.0,
        db_disturbance_margin_pct=5.0,
        queue_disturbance_margin_pct=3.0,
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def run_simulation(
    duration_seconds: float = 120.0,
) -> Generator[Tuple[WorkloadRequest, TelemetrySnapshot, str], None, None]:
    """Yield (WorkloadRequest, TelemetrySnapshot, phase_name) at ~2-second intervals.

    The generator drives a scripted 5-phase narrative arc:
      nominal → escalation → spike → recovery → steady

    Parameters
    ----------
    duration_seconds :
        Total simulated wall-clock duration. Each phase gets a proportional
        slice. Minimum effective step interval is ``_STEP_INTERVAL_S`` seconds.
    """
    # Build phase time-spans
    phases: list[Tuple[str, float, float]] = []  # (name, start_s, end_s)
    cursor = 0.0
    for name, fraction in _PHASES:
        duration = duration_seconds * fraction
        phases.append((name, cursor, cursor + duration))
        cursor += duration

    start_wall = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_wall
        if elapsed >= duration_seconds:
            break

        # Determine current phase
        current_phase = "steady"
        t_norm = 1.0
        for name, t_start, t_end in phases:
            if elapsed < t_end:
                current_phase = name
                span = t_end - t_start
                t_norm = (elapsed - t_start) / span if span > 0 else 0.0
                t_norm = _clamp(t_norm, 0.0, 1.0)
                break

        profile_fn = _PROFILE_FNS[current_phase]
        gpu, api, db, q, err = profile_fn(t_norm)

        request  = _workload_for_phase(current_phase, t_norm)
        snapshot = _build_snapshot(gpu, api, db, q, err)

        yield request, snapshot, current_phase

        # Sleep until next step
        if _STEP_INTERVAL_S > 0:
            next_step = start_wall + (math.floor(elapsed / _STEP_INTERVAL_S) + 1) * _STEP_INTERVAL_S
            sleep_s = next_step - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
