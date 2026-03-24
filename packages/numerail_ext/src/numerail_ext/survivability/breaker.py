from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .types import BreakerDecision, BreakerMode, BreakerThresholds, TelemetrySnapshot


@dataclass
class BreakerStateMachine:
    """Hysteretic state machine for supervisory mode selection.

    Thread-safe.  All mode reads and transitions are serialized by an
    internal lock.  External callers must use :meth:`force_mode` rather
    than assigning ``mode`` directly.
    """

    thresholds: BreakerThresholds
    mode: BreakerMode = BreakerMode.CLOSED
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── score ────────────────────────────────────────────────────────

    def overload_score(self, snapshot: TelemetrySnapshot) -> float:
        """Compute a bounded [0, 1] operational stress score from trusted telemetry."""
        error_term = min(1.0, max(0.0, snapshot.current_error_rate_pct) / 10.0)
        return (
            0.30 * snapshot.current_gpu_util
            + 0.25 * snapshot.current_api_util
            + 0.20 * snapshot.current_db_util
            + 0.10 * snapshot.current_queue_util
            + 0.15 * error_term
        )

    # ── transitions ──────────────────────────────────────────────────

    def update(self, snapshot: TelemetrySnapshot) -> BreakerDecision:
        """Evaluate telemetry and transition mode.  Thread-safe."""
        with self._lock:
            score = self.overload_score(snapshot)

            if score >= self.thresholds.safe_stop_score:
                self.mode = BreakerMode.SAFE_STOP
                return BreakerDecision(self.mode, score, "safe-stop threshold crossed")

            if self.mode == BreakerMode.CLOSED:
                if score >= self.thresholds.trip_score:
                    self.mode = BreakerMode.THROTTLED
                    return BreakerDecision(self.mode, score, "trip threshold crossed")
                return BreakerDecision(self.mode, score, "normal")

            if self.mode == BreakerMode.THROTTLED:
                if score <= self.thresholds.reset_score:
                    self.mode = BreakerMode.HALF_OPEN
                    return BreakerDecision(self.mode, score, "recovery threshold crossed")
                return BreakerDecision(self.mode, score, "remain throttled")

            if self.mode == BreakerMode.HALF_OPEN:
                if score >= self.thresholds.trip_score:
                    self.mode = BreakerMode.THROTTLED
                    return BreakerDecision(self.mode, score, "re-trip during half-open")
                if score <= self.thresholds.reset_score:
                    self.mode = BreakerMode.CLOSED
                    return BreakerDecision(self.mode, score, "fully recovered")
                return BreakerDecision(self.mode, score, "remain half-open")

            if self.mode == BreakerMode.SAFE_STOP:
                return BreakerDecision(self.mode, score, "safe-stop latched")

            # OPEN — defensive catch-all
            return BreakerDecision(self.mode, score, "open")

    def force_mode(self, mode: BreakerMode) -> None:
        """Force a mode transition from outside the state machine.  Thread-safe."""
        with self._lock:
            self.mode = mode

    def reset(self, snapshot: TelemetrySnapshot) -> BreakerDecision:
        """Manual recovery from any latched state.

        Only transitions to CLOSED if the overload score is at or below
        the reset threshold.  Returns a :class:`BreakerDecision` describing
        the outcome.
        """
        with self._lock:
            score = self.overload_score(snapshot)
            if score > self.thresholds.reset_score:
                return BreakerDecision(
                    self.mode, score,
                    "reset denied: overload score above reset threshold",
                )
            self.mode = BreakerMode.CLOSED
            return BreakerDecision(self.mode, score, "manual reset to closed")
