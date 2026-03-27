"""EnforcementExperience and EnforcementExperienceBuffer.

Captures full-context enforcement decisions for LLM training.
Thread-safe circular buffer with filtering, sampling, and persistence.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EnforcementExperience
# ---------------------------------------------------------------------------


@dataclass
class EnforcementExperience:
    """A single enforcement event captured for training.

    Fields
    ------
    experience_id : str
        Unique identifier (``exp-000001`` format).
    action_id : str
        The action ID from the enforcement engine.
    timestamp_ms : float
        Unix time in milliseconds at capture.

    conversation_context : List[Dict[str, Any]]
        Message history leading to this tool call.
    tool_call : Dict[str, Any]
        Raw tool call ``{"name": str, "arguments": dict}``.

    proposed_vector : np.ndarray
        The quantified action proposal seen by the enforcement engine.

    result : str
        ``"approve"``, ``"project"``, or ``"reject"``.
    enforced_vector : Optional[np.ndarray]
        The corrected vector; ``None`` for REJECT.
    distance : float
        0.0 for APPROVE, >0 for PROJECT, -1.0 for REJECT.
    violations : List[Tuple[str, float]]
        ``(constraint_name, violation_magnitude)`` pairs.
    solver_method : str
        Solver that produced the decision.
    routing_decision : Optional[str]
        Routing annotation if present.

    breaker_mode : str
        Breaker state at time of enforcement.
    budget_remaining : Dict[str, float]
        Budget balances at time of enforcement.
    overload_score : float
        Overload score used by the breaker.
    policy_digest : str
        SHA-256 digest of the active policy configuration.

    reward : float
        Shaped scalar reward (populated by reward shaper).
    reward_components : Optional[Dict[str, float]]
        Breakdown of reward components.
    dimension_feedback : Optional[Dict[str, float]]
        Per-dimension signed deltas ``proposed - enforced`` for PROJECT.
    """

    # Identity
    experience_id: str
    action_id: str
    timestamp_ms: float

    # Context
    conversation_context: List[Dict[str, Any]]
    tool_call: Dict[str, Any]

    # Quantified proposal
    proposed_vector: np.ndarray

    # Enforcement decision
    result: str
    enforced_vector: Optional[np.ndarray]
    distance: float
    violations: List[Tuple[str, float]]
    solver_method: str
    routing_decision: Optional[str]

    # Operational context
    breaker_mode: str
    budget_remaining: Dict[str, float]
    overload_score: float
    policy_digest: str

    # Training signal
    reward: float = 0.0
    reward_components: Optional[Dict[str, float]] = None
    dimension_feedback: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_from_output(enforcement_output: Any) -> Dict[str, Any]:
    """Extract common fields from EnforcementOutput or GovernedStep."""
    # GovernedStep (numerail_ext): has .numerail_result, .breaker, .grant
    if hasattr(enforcement_output, "numerail_result"):
        nr = enforcement_output.numerail_result or {}
        result_raw = nr.get("decision", "reject")
        action_id  = nr.get("action_id", "")
        enforced   = nr.get("enforced_values")  # dict or None
        feedback   = nr.get("feedback", [])
        solver     = nr.get("solver_method", "")
        routing    = nr.get("routing_decision", None)

        # Violations: feedback items may be strings or (name, mag) tuples
        violations: List[Tuple[str, float]] = []
        if isinstance(feedback, list):
            for item in feedback:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    violations.append((str(item[0]), float(item[1])))
                elif isinstance(item, str):
                    violations.append((item, 1.0))

        # Distance: use enforced vs proposed magnitude if available
        distance_val = float(nr.get("distance", -1.0))
        if distance_val == -1.0:
            result_lower = result_raw.lower()
            if result_lower == "approve":
                distance_val = 0.0
            elif result_lower == "reject":
                distance_val = -1.0

        return {
            "result":    result_raw.lower(),
            "action_id": action_id,
            "enforced_dict": enforced,
            "violations": violations,
            "solver_method": solver,
            "routing_decision": routing,
            "distance": distance_val,
        }

    # Plain dict (raw service output)
    if isinstance(enforcement_output, dict):
        result_raw = enforcement_output.get("decision", "reject")
        # Parse violations from feedback field (same format as GovernedStep path)
        violations: List[Tuple[str, float]] = []
        for item in enforcement_output.get("feedback", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                violations.append((str(item[0]), float(item[1])))
            elif isinstance(item, str):
                violations.append((item, 1.0))
        return {
            "result":    result_raw.lower(),
            "action_id": enforcement_output.get("action_id", ""),
            "enforced_dict": enforcement_output.get("enforced_values"),
            "violations": violations,
            "solver_method": enforcement_output.get("solver_method", ""),
            "routing_decision": enforcement_output.get("routing_decision"),
            "distance": float(enforcement_output.get("distance", -1.0)),
        }

    return {
        "result": "reject", "action_id": "", "enforced_dict": None,
        "violations": [], "solver_method": "", "routing_decision": None,
        "distance": -1.0,
    }


def _dict_to_array(d: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not d:
        return None
    return np.array(list(d.values()), dtype=np.float64)


# ---------------------------------------------------------------------------
# EnforcementExperienceBuffer
# ---------------------------------------------------------------------------


class EnforcementExperienceBuffer:
    """Thread-safe circular buffer of enforcement experiences.

    Fixed capacity; oldest experiences are evicted when full.

    Parameters
    ----------
    max_size :
        Maximum number of experiences to retain. Defaults to 100,000.
    """

    def __init__(self, max_size: int = 100_000) -> None:
        self._max_size = max_size
        self._buffer: Deque[EnforcementExperience] = deque()
        self._lock = threading.Lock()
        self._counter = 0
        # Per-result fast counters (maintained alongside _buffer)
        self._approve_count = 0
        self._project_count = 0
        self._reject_count  = 0

    # ── Recording ──────────────────────────────────────────────────────────

    def record(
        self,
        conversation_context: List[Dict[str, Any]],
        tool_call: Dict[str, Any],
        proposed_vector: np.ndarray,
        enforcement_output: Any,
        enforced_vector: Optional[np.ndarray] = None,
        breaker_mode: str = "closed",
        budget_remaining: Optional[Dict[str, float]] = None,
        overload_score: float = 0.0,
        policy_digest: str = "",
        reward: float = 0.0,
        reward_components: Optional[Dict[str, float]] = None,
        dimension_feedback: Optional[Dict[str, float]] = None,
    ) -> EnforcementExperience:
        """Record an enforcement experience. Returns the created experience."""
        extracted = _extract_from_output(enforcement_output)
        result       = extracted["result"]
        action_id    = extracted["action_id"] or f"action-{uuid.uuid4().hex[:8]}"
        violations   = extracted["violations"]
        solver       = extracted["solver_method"]
        routing      = extracted["routing_decision"]
        distance_val = extracted["distance"]

        # Compute enforced_vector if not provided
        if enforced_vector is None and extracted["enforced_dict"] is not None:
            enforced_vector = _dict_to_array(extracted["enforced_dict"])

        # Distance refinement: compute from vectors if not already set
        if distance_val == -1.0 and result in ("approve", "project") and enforced_vector is not None:
            distance_val = float(np.linalg.norm(proposed_vector - enforced_vector))
        if result == "approve":
            distance_val = 0.0
        if result == "reject" and distance_val >= 0.0:
            distance_val = -1.0

        with self._lock:
            self._counter += 1
            exp_id = f"exp-{self._counter:06d}"
            ts_ms  = float(time.time_ns() // 1_000_000)

            exp = EnforcementExperience(
                experience_id=exp_id,
                action_id=action_id,
                timestamp_ms=ts_ms,
                conversation_context=list(conversation_context),
                tool_call=dict(tool_call),
                proposed_vector=np.array(proposed_vector, dtype=np.float64),
                result=result,
                enforced_vector=(np.array(enforced_vector, dtype=np.float64)
                                 if enforced_vector is not None else None),
                distance=distance_val,
                violations=list(violations),
                solver_method=solver,
                routing_decision=routing,
                breaker_mode=breaker_mode,
                budget_remaining=dict(budget_remaining or {}),
                overload_score=overload_score,
                policy_digest=policy_digest,
                reward=reward,
                reward_components=reward_components,
                dimension_feedback=dimension_feedback,
            )

            # Evict oldest if at capacity
            if len(self._buffer) >= self._max_size:
                evicted = self._buffer.popleft()
                self._dec_counter(evicted.result)

            self._buffer.append(exp)
            self._inc_counter(result)

        return exp

    # ── Internal counters ──────────────────────────────────────────────────

    def _inc_counter(self, result: str) -> None:
        if result == "approve":   self._approve_count += 1
        elif result == "project": self._project_count += 1
        else:                     self._reject_count  += 1

    def _dec_counter(self, result: str) -> None:
        if result == "approve":   self._approve_count = max(0, self._approve_count - 1)
        elif result == "project": self._project_count = max(0, self._project_count - 1)
        else:                     self._reject_count  = max(0, self._reject_count  - 1)

    # ── Size & counts ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def size(self) -> int:
        return len(self)

    @property
    def approve_count(self) -> int:
        with self._lock:
            return self._approve_count

    @property
    def project_count(self) -> int:
        with self._lock:
            return self._project_count

    @property
    def reject_count(self) -> int:
        with self._lock:
            return self._reject_count

    @property
    def approval_rate(self) -> float:
        with self._lock:
            total = len(self._buffer)
            return self._approve_count / total if total > 0 else 0.0

    # ── Retrieval ──────────────────────────────────────────────────────────

    def get_all(self, filter_result: Optional[str] = None) -> List[EnforcementExperience]:
        """Return all experiences, optionally filtered by result."""
        with self._lock:
            items = list(self._buffer)
        if filter_result is not None:
            filt = filter_result.lower()
            items = [e for e in items if e.result == filt]
        return items

    def get_project_experiences(self) -> List[EnforcementExperience]:
        """Shorthand for ``get_all(filter_result='project')``."""
        return self.get_all(filter_result="project")

    def sample_batch(
        self,
        batch_size: int,
        filter_result: Optional[str] = None,
    ) -> List[EnforcementExperience]:
        """Random sample without replacement from the buffer."""
        pool = self.get_all(filter_result=filter_result)
        if not pool:
            return []
        rng = np.random.default_rng()
        n   = min(batch_size, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    def get_approve_reject_pairs(
        self,
        max_pairs: int = 1000,
    ) -> List[Tuple[EnforcementExperience, EnforcementExperience]]:
        """Find (preferred=APPROVE, dispreferred=REJECT) pairs.

        Pairs are matched on ``breaker_mode``. For each REJECT experience
        the nearest APPROVE experience by timestamp in the same mode is used.
        """
        with self._lock:
            items = list(self._buffer)

        approves = [e for e in items if e.result == "approve"]
        rejects  = [e for e in items if e.result == "reject"]

        pairs: List[Tuple[EnforcementExperience, EnforcementExperience]] = []
        for rej in rejects:
            candidates = [a for a in approves if a.breaker_mode == rej.breaker_mode]
            if not candidates:
                continue
            # Nearest by timestamp
            best = min(candidates, key=lambda a: abs(a.timestamp_ms - rej.timestamp_ms))
            pairs.append((best, rej))
            if len(pairs) >= max_pairs:
                break

        return pairs

    # ── Analytics ──────────────────────────────────────────────────────────

    def dimension_violation_frequency(self) -> Dict[str, int]:
        """Count appearances of each constraint name across all violations."""
        freq: Dict[str, int] = {}
        with self._lock:
            items = list(self._buffer)
        for exp in items:
            for name, _ in exp.violations:
                freq[name] = freq.get(name, 0) + 1
        return freq

    def mean_distance_by_result(self) -> Dict[str, float]:
        """Mean enforcement distance grouped by result type."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int]   = {}
        with self._lock:
            items = list(self._buffer)
        for exp in items:
            r = exp.result
            d = exp.distance if exp.distance >= 0 else 0.0
            totals[r] = totals.get(r, 0.0) + d
            counts[r] = counts.get(r, 0) + 1
        return {r: totals[r] / counts[r] for r in totals}

    # ── Persistence ────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        """Write all experiences to a JSON file."""
        with self._lock:
            items = list(self._buffer)

        def _serialize(exp: EnforcementExperience) -> dict:
            return {
                "experience_id":        exp.experience_id,
                "action_id":            exp.action_id,
                "timestamp_ms":         exp.timestamp_ms,
                "conversation_context": exp.conversation_context,
                "tool_call":            exp.tool_call,
                "proposed_vector":      exp.proposed_vector.tolist(),
                "result":               exp.result,
                "enforced_vector":      (exp.enforced_vector.tolist()
                                         if exp.enforced_vector is not None else None),
                "distance":             exp.distance,
                "violations":           exp.violations,
                "solver_method":        exp.solver_method,
                "routing_decision":     exp.routing_decision,
                "breaker_mode":         exp.breaker_mode,
                "budget_remaining":     exp.budget_remaining,
                "overload_score":       exp.overload_score,
                "policy_digest":        exp.policy_digest,
                "reward":               exp.reward,
                "reward_components":    exp.reward_components,
                "dimension_feedback":   exp.dimension_feedback,
            }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump([_serialize(e) for e in items], fh)

    def import_json(self, path: str) -> int:
        """Load experiences from a JSON file. Returns count imported."""
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)

        count = 0
        for row in rows:
            ev_raw = row.get("enforced_vector")
            ev_arr = np.array(ev_raw, dtype=np.float64) if ev_raw is not None else None

            exp = EnforcementExperience(
                experience_id=      row["experience_id"],
                action_id=          row["action_id"],
                timestamp_ms=       float(row["timestamp_ms"]),
                conversation_context=row.get("conversation_context", []),
                tool_call=          row.get("tool_call", {}),
                proposed_vector=    np.array(row["proposed_vector"], dtype=np.float64),
                result=             row["result"],
                enforced_vector=    ev_arr,
                distance=           float(row["distance"]),
                violations=         [tuple(v) for v in row.get("violations", [])],
                solver_method=      row.get("solver_method", ""),
                routing_decision=   row.get("routing_decision"),
                breaker_mode=       row.get("breaker_mode", "closed"),
                budget_remaining=   row.get("budget_remaining", {}),
                overload_score=     float(row.get("overload_score", 0.0)),
                policy_digest=      row.get("policy_digest", ""),
                reward=             float(row.get("reward", 0.0)),
                reward_components=  row.get("reward_components"),
                dimension_feedback= row.get("dimension_feedback"),
            )
            with self._lock:
                if len(self._buffer) >= self._max_size:
                    evicted = self._buffer.popleft()
                    self._dec_counter(evicted.result)
                self._buffer.append(exp)
                self._inc_counter(exp.result)
            count += 1

        return count

    def clear(self) -> None:
        """Remove all experiences."""
        with self._lock:
            self._buffer.clear()
            self._approve_count = 0
            self._project_count = 0
            self._reject_count  = 0
