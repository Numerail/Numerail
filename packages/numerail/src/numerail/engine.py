"""
Numerail v5.0.0 — Deterministic Geometric Enforcement for AI Actuation Safety

Deterministic enforcement of convex constraints on AI numerical outputs.
Supports linear (Ax ≤ b), quadratic (x'Qx + a'x ≤ b), second-order cone
(‖Mx + q‖ ≤ c'x + d), and positive semidefinite (A₀ + Σ xᵢAᵢ ≽ 0) constraints
in any combination.

THE GUARANTEE (Theorem 1):
    If enforce() returns APPROVE or PROJECT, the enforced vector satisfies every
    active constraint. This holds for ALL inputs, ALL models, ALL solvers.
    The post-check is the trust boundary. The solver is untrusted.

    Proved in proof/PROOF.md. Verified by verify_proof.py (3,732 checks)
    and test_guarantee.py (45 tests across 7 categories).

Architecture:
    ConvexConstraint (ABC)   — evaluate / is_satisfied / project_hint
    FeasibleRegion           — intersection of ConvexConstraints
    enforce()                — THE TRUST BOUNDARY: check → approve | solve → check → project | reject
    NumerailSystem           — full integration: schema + enforcement + budget + audit + metrics

Dependencies: numpy, scipy

    enforce(vector, region) → APPROVE | PROJECT | REJECT
"""

from __future__ import annotations

import json
import hashlib
import threading
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, linprog


# ═══════════════════════════════════════════════════════════════════════════
#  EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════

class NumerailError(Exception):
    """Base exception for all Numerail errors."""


class ValidationError(NumerailError):
    """Malformed inputs: NaN, Inf, dimension mismatch, bad mode string."""


class ConstraintError(NumerailError):
    """Invalid constraint specification: non-PSD Q, zero weights, shape mismatch."""


class InfeasibleRegionError(NumerailError):
    """The feasible region is empty — no point satisfies all constraints."""


class SolverError(NumerailError):
    """All solvers in the fallback chain failed. Informational only; enforce returns REJECT."""


class SchemaError(NumerailError):
    """Schema mismatch: missing field, duplicate name, normalizer on unknown field."""


class ResolutionError(NumerailError):
    """Constraint name not found in the active region. Distinct from ValidationError."""


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _validate_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or Inf")


def _validate_vector(x: ArrayLike, expected_dim: int, name: str) -> np.ndarray:
    """Coerce to float64 1-D vector, validate dimension and finiteness."""
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.shape[0] != expected_dim:
        raise ValidationError(f"{name}: expected dimension {expected_dim}, got {v.shape[0]}")
    _validate_finite(v, name)
    return v


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deterministic_json(obj: Any) -> str:
    """Canonical JSON for hashing. Sorted keys, no whitespace, numpy-safe."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class EnforcementResult(Enum):
    """The three possible outcomes. Definition 5.1."""
    APPROVE = "approve"
    PROJECT = "project"
    REJECT = "reject"


class DimensionPolicy(Enum):
    """Per-dimension projection control."""
    FREELY_PROJECTABLE = "freely_projectable"
    PROJECT_WITH_FLAG = "project_with_flag"
    PROJECTION_FORBIDDEN = "forbidden"


class RoutingDecision(Enum):
    """Maps correction distance to human-attention tiers."""
    SILENT_PROJECT = "silent_project"
    FLAGGED_PROJECT = "flagged_project"
    CONFIRMATION_REQUIRED = "confirmation_required"
    HARD_REJECT = "hard_reject"


@dataclass(frozen=True)
class RoutingThresholds:
    """Distance thresholds for routing decisions. All must be non-negative
    and monotonically ordered: 0 ≤ silent ≤ flagged ≤ confirmation ≤ hard_reject."""
    silent: float = 0.05
    flagged: float = 0.20
    confirmation: float = 0.40
    hard_reject: float = 0.80

    def __post_init__(self):
        vals = (self.silent, self.flagged, self.confirmation, self.hard_reject)
        for name, v in zip(("silent", "flagged", "confirmation", "hard_reject"), vals):
            if v < 0:
                raise ValidationError(
                    f"RoutingThresholds.{name} must be non-negative, got {v}"
                )
        if not (self.silent <= self.flagged <= self.confirmation <= self.hard_reject):
            raise ValidationError(
                f"RoutingThresholds must satisfy silent ≤ flagged ≤ confirmation "
                f"≤ hard_reject, got {vals}"
            )


def _compute_routing(distance: float, thresholds: RoutingThresholds) -> RoutingDecision:
    """Map projection distance to routing tier."""
    if distance <= thresholds.silent:
        return RoutingDecision.SILENT_PROJECT
    elif distance <= thresholds.flagged:
        return RoutingDecision.FLAGGED_PROJECT
    elif distance <= thresholds.confirmation:
        return RoutingDecision.CONFIRMATION_REQUIRED
    else:
        return RoutingDecision.HARD_REJECT


# ═══════════════════════════════════════════════════════════════════════════
#  CONVEX CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════


class ConvexConstraint(ABC):
    """
    Abstract base for any convex constraint.

    Contract:
        evaluate(x) ≤ 0  ⟺  x satisfies this constraint
        is_satisfied(x, tol)  ⟺  evaluate(x) ≤ tol

    The CorrectChecker property (Axiom 1):
        ∀ P x, check P x = true ↔ P x
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, unique within a FeasibleRegion."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected input vector length."""
        ...

    @property
    def tag(self) -> str:
        """Metadata tag (e.g., source, category, authority). Default empty."""
        return ""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Violation magnitude. Negative or zero = satisfied, positive = violated."""
        ...

    def is_satisfied(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """THE POST-CHECK for this constraint. The trust boundary."""
        return self.evaluate(x) <= tol

    def violation(self, x: np.ndarray) -> float:
        """Positive violation magnitude, or 0.0 if satisfied."""
        return max(0.0, self.evaluate(x))

    def project_hint(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Best-effort single-constraint projection. Override in subclasses with
        closed-form or efficient projectors. Returns None if unavailable.
        """
        return None

    @property
    def constraint_names(self) -> Tuple[str, ...]:
        """Names of sub-constraints within this constraint."""
        return (self.name,)


# ═══════════════════════════════════════════════════════════════════════════
#  LINEAR CONSTRAINTS — Ax ≤ b
# ═══════════════════════════════════════════════════════════════════════════


class LinearConstraints(ConvexConstraint):
    """Ax ≤ b — a convex polytope defined by m linear inequalities in n dimensions."""

    def __init__(
        self,
        A: ArrayLike,
        b: ArrayLike,
        names: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        constraint_name: str = "linear",
    ):
        self._A = np.array(A, dtype=np.float64)
        self._b = np.array(b, dtype=np.float64).ravel()
        if self._A.ndim != 2:
            raise ConstraintError(f"A must be 2-D, got {self._A.ndim}-D")
        m, n = self._A.shape
        if self._b.shape[0] != m:
            raise ConstraintError(f"A has {m} rows, b has {self._b.shape[0]} entries")
        _validate_finite(self._A, "A")
        _validate_finite(self._b, "b")

        self._row_names: Tuple[str, ...] = (
            tuple(names) if names else tuple(f"c_{i}" for i in range(m))
        )
        if len(self._row_names) != m:
            raise ConstraintError(
                f"names length {len(self._row_names)} != constraint count {m}"
            )
        if len(set(self._row_names)) != len(self._row_names):
            seen = set()
            for rn in self._row_names:
                if rn in seen:
                    raise ConstraintError(
                        f"Duplicate row name '{rn}' within LinearConstraints block"
                    )
                seen.add(rn)
        self._row_tags: Tuple[str, ...] = (
            tuple(tags) if tags else ("",) * m
        )
        if len(self._row_tags) != m:
            raise ConstraintError(
                f"tags length {len(self._row_tags)} != constraint count {m}"
            )
        self._constraint_name = constraint_name
        self._m = m
        self._n = n
        self._norms_sq = np.sum(self._A * self._A, axis=1)
        self._norms_sq = np.where(self._norms_sq < 1e-30, 1e-30, self._norms_sq)

    # ── ConvexConstraint ABC ──

    @property
    def name(self) -> str:
        return self._constraint_name

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def tag(self) -> str:
        for a in self._row_tags:
            if a:
                return a
        return ""

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.max(self._A @ x - self._b))

    def is_satisfied(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(np.all(self._A @ x <= self._b + tol))

    def project_hint(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Iterative half-space projection onto most-violated row."""
        y = x.copy()
        for _ in range(500):
            residuals = self._A @ y - self._b
            violated = np.where(residuals > 1e-12)[0]
            if len(violated) == 0:
                break
            worst = violated[np.argmax(residuals[violated])]
            y -= (residuals[worst] / self._norms_sq[worst]) * self._A[worst]
        return y

    @property
    def constraint_names(self) -> Tuple[str, ...]:
        return self._row_names

    # ── Matrix access ──

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def row_names(self) -> Tuple[str, ...]:
        return self._row_names

    @property
    def row_tags(self) -> Tuple[str, ...]:
        return self._row_tags

    @property
    def n_rows(self) -> int:
        return self._m

    # ── Per-row diagnostics ──

    def row_violations(self, x: np.ndarray) -> Dict[str, float]:
        residuals = self._A @ x - self._b
        return {
            self._row_names[i]: float(residuals[i])
            for i in range(self._m)
            if residuals[i] > 1e-9
        }

    def row_slack(self, x: np.ndarray) -> np.ndarray:
        return self._b - self._A @ x

    def row_bindings(self, x: np.ndarray, tol: float = 1e-6) -> List[str]:
        slack = self._b - self._A @ x
        return [self._row_names[i] for i in range(self._m) if abs(slack[i]) < tol]

    # ── Copy-on-modify ──

    def with_bound(self, row_name: str, new_bound: float) -> LinearConstraints:
        for i, n in enumerate(self._row_names):
            if n == row_name:
                new_b = self._b.copy()
                new_b[i] = new_bound
                return LinearConstraints(
                    self._A.copy(), new_b, self._row_names, self._row_tags,
                    self._constraint_name,
                )
        raise ResolutionError(f"Row name '{row_name}' not found in {self._constraint_name}")

    def with_bounds_by_index(self, updates: Dict[int, float]) -> LinearConstraints:
        new_b = self._b.copy()
        for idx, val in updates.items():
            if not (0 <= idx < self._m):
                raise ValidationError(f"Row index {idx} out of range [0, {self._m})")
            new_b[idx] = val
        return LinearConstraints(
            self._A.copy(), new_b, self._row_names, self._row_tags,
            self._constraint_name,
        )

    def with_safety_margin(self, margin: float) -> LinearConstraints:
        """Return copy with bounds tightened by multiplicative margin ∈ (0, 1].
        For upper bounds (b ≥ 0): new_b = b * margin (shrinks toward zero).
        For lower bounds (b < 0): new_b = b / margin (moves further from zero,
        i.e. more negative, which tightens the lower bound inward).
        Both directions contract the feasible region."""
        if margin <= 0 or margin > 1.0:
            raise ValidationError(f"safety_margin must be in (0, 1], got {margin}")
        new_b = np.where(self._b >= 0, self._b * margin, self._b / margin)
        return LinearConstraints(
            self._A.copy(), new_b, self._row_names,
            self._row_tags, self._constraint_name,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  QUADRATIC CONSTRAINT — x'Qx + a'x ≤ b
# ═══════════════════════════════════════════════════════════════════════════


class QuadraticConstraint(ConvexConstraint):
    """x'Qx + a'x ≤ b — ellipsoidal constraint. Q must be PSD."""

    def __init__(
        self,
        Q: ArrayLike,
        a: ArrayLike,
        b: float,
        constraint_name: str = "quadratic",
        constraint_tag: str = "",
    ):
        self._Q = np.asarray(Q, dtype=np.float64)
        self._a = np.asarray(a, dtype=np.float64).ravel()
        self._b = float(b)
        if self._Q.ndim != 2 or self._Q.shape[0] != self._Q.shape[1]:
            raise ConstraintError("Q must be square")
        n = self._Q.shape[0]
        if self._a.shape[0] != n:
            raise ConstraintError(f"a has dim {self._a.shape[0]}, Q has dim {n}")
        _validate_finite(self._Q, "Q")
        _validate_finite(self._a, "a")
        if not np.isfinite(self._b):
            raise ConstraintError("b is not finite")
        self._Q = 0.5 * (self._Q + self._Q.T)
        eigvals = np.linalg.eigvalsh(self._Q)
        if np.min(eigvals) < -1e-8:
            raise ConstraintError(
                f"Q is not PSD: min eigenvalue = {np.min(eigvals):.2e}. "
                "Quadratic constraint requires Q ≽ 0 for convexity."
            )
        self._name = constraint_name
        self._tag = constraint_tag
        self._n = n

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @property
    def linear_term(self) -> np.ndarray:
        return self._a

    @property
    def bound(self) -> float:
        return self._b

    def evaluate(self, x: np.ndarray) -> float:
        return float(x @ self._Q @ x + self._a @ x - self._b)

    def project_hint(self, x: np.ndarray) -> Optional[np.ndarray]:
        """KKT bisection on the dual variable λ."""
        if self.is_satisfied(x, tol=0.0):
            return x.copy()
        Q, a_vec, bound, n = self._Q, self._a, self._b, self._n

        def _solve(lam):
            M = np.eye(n) + 2.0 * lam * Q
            try:
                return np.linalg.solve(M, x - lam * a_vec)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(M, x - lam * a_vec, rcond=None)[0]

        def _violation(lam):
            y = _solve(lam)
            return float(y @ Q @ y + a_vec @ y - bound)

        lo, hi = 0.0, 1.0
        for _ in range(50):
            if _violation(hi) <= 0:
                break
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _violation(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return _solve(hi)


# ═══════════════════════════════════════════════════════════════════════════
#  SOCP CONSTRAINT — ‖Mx + q‖ ≤ c'x + d
# ═══════════════════════════════════════════════════════════════════════════


class SOCPConstraint(ConvexConstraint):
    """‖Mx + q‖ ≤ c'x + d — second-order cone constraint."""

    def __init__(
        self,
        M: ArrayLike,
        q: ArrayLike,
        c: ArrayLike,
        d: float,
        constraint_name: str = "socp",
        constraint_tag: str = "",
    ):
        self._M = np.asarray(M, dtype=np.float64)
        self._q = np.asarray(q, dtype=np.float64).ravel()
        self._c = np.asarray(c, dtype=np.float64).ravel()
        self._d = float(d)
        if self._M.ndim != 2:
            raise ConstraintError("M must be 2-D")
        m, n = self._M.shape
        if self._q.shape[0] != m:
            raise ConstraintError(f"q has dim {self._q.shape[0]}, M has {m} rows")
        if self._c.shape[0] != n:
            raise ConstraintError(f"c has dim {self._c.shape[0]}, M has {n} cols")
        _validate_finite(self._M, "M")
        _validate_finite(self._q, "q")
        _validate_finite(self._c, "c")
        if not np.isfinite(self._d):
            raise ConstraintError("d is not finite")
        self._name = constraint_name
        self._tag = constraint_tag
        self._n = n

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def tag(self) -> str:
        return self._tag

    def evaluate(self, x: np.ndarray) -> float:
        lhs = float(np.linalg.norm(self._M @ x + self._q))
        rhs = float(self._c @ x + self._d)
        return lhs - rhs


# ═══════════════════════════════════════════════════════════════════════════
#  PSD CONSTRAINT — A₀ + Σ xᵢAᵢ ≽ 0
# ═══════════════════════════════════════════════════════════════════════════


class PSDConstraint(ConvexConstraint):
    """A₀ + Σᵢ xᵢAᵢ ≽ 0 — positive semidefinite (LMI) constraint."""

    def __init__(
        self,
        A0: ArrayLike,
        A_list: Sequence[ArrayLike],
        constraint_name: str = "psd",
        constraint_tag: str = "",
    ):
        self._A0 = np.asarray(A0, dtype=np.float64)
        if self._A0.ndim != 2 or self._A0.shape[0] != self._A0.shape[1]:
            raise ConstraintError("A0 must be square")
        k = self._A0.shape[0]
        _validate_finite(self._A0, "A0")
        self._A0 = 0.5 * (self._A0 + self._A0.T)

        self._A_list: List[np.ndarray] = []
        for i, Ai in enumerate(A_list):
            arr = np.asarray(Ai, dtype=np.float64)
            if arr.shape != (k, k):
                raise ConstraintError(f"A_{i} shape {arr.shape}, expected ({k},{k})")
            _validate_finite(arr, f"A_{i}")
            self._A_list.append(0.5 * (arr + arr.T))

        self._name = constraint_name
        self._tag = constraint_tag
        self._n = len(A_list)
        self._k = k

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def matrix_size(self) -> int:
        return self._k

    def matrix_at(self, x: np.ndarray) -> np.ndarray:
        result = self._A0.copy()
        for i, Ai in enumerate(self._A_list):
            result += x[i] * Ai
        return result

    def evaluate(self, x: np.ndarray) -> float:
        eigvals = np.linalg.eigvalsh(self.matrix_at(x))
        return float(-np.min(eigvals))


# ═══════════════════════════════════════════════════════════════════════════
#  FEASIBLE REGION
# ═══════════════════════════════════════════════════════════════════════════

class FeasibleRegion:
    """Intersection of convex constraints of potentially different types."""

    def __init__(self, constraints: List[ConvexConstraint], n_dim: int):
        if not constraints:
            raise ConstraintError("Must provide at least one constraint")
        for c in constraints:
            if c.dimension != n_dim:
                raise ConstraintError(
                    f"Constraint '{c.name}' has dimension {c.dimension}, expected {n_dim}"
                )
        self._constraints = list(constraints)
        self._n = n_dim
        self._version = ""
        self._name_index: Dict[str, Tuple[int, Optional[int]]] = {}
        for ci, c in enumerate(self._constraints):
            for ri, rn in enumerate(c.constraint_names):
                if rn in self._name_index:
                    prev_ci, prev_ri = self._name_index[rn]
                    raise ConstraintError(
                        f"Duplicate constraint name '{rn}' "
                        f"(constraint blocks {prev_ci} and {ci}). "
                        "Names must be globally unique within a FeasibleRegion."
                    )
                self._name_index[rn] = (ci, ri if len(c.constraint_names) > 1 else None)

    @property
    def constraints(self) -> List[ConvexConstraint]:
        return list(self._constraints)

    @property
    def n_dim(self) -> int:
        return self._n

    @property
    def version(self) -> str:
        return self._version

    def with_version(self, version: str) -> FeasibleRegion:
        r = FeasibleRegion(self._constraints, self._n)
        r._version = version
        return r

    # ── THE COMBINED POST-CHECK ──

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """THE TRUST BOUNDARY. True iff x satisfies ALL constraints."""
        return all(c.is_satisfied(x, tol) for c in self._constraints)

    # ── Diagnostics ──

    def violations(self, x: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for c in self._constraints:
            if isinstance(c, LinearConstraints):
                for name, mag in c.row_violations(x).items():
                    if mag > tol:
                        result[name] = mag
            else:
                v = c.violation(x)
                if v > tol:
                    result[c.name] = v
        return result

    def violated_names(self, x: np.ndarray, tol: float = 1e-6) -> Tuple[str, ...]:
        return tuple(self.violations(x, tol).keys())

    def binding_names(self, x: np.ndarray, tol: float = 1e-6) -> Tuple[str, ...]:
        result: List[str] = []
        for c in self._constraints:
            if isinstance(c, LinearConstraints):
                result.extend(c.row_bindings(x, tol))
            elif abs(c.evaluate(x)) <= tol:
                result.append(c.name)
        return tuple(result)

    # ── Name resolution ──

    def resolve_name(self, name: str) -> Tuple[int, Optional[int]]:
        if name not in self._name_index:
            available = list(self._name_index.keys())
            raise ResolutionError(
                f"Constraint name '{name}' not found. Available: {available}"
            )
        return self._name_index[name]

    def resolve_names(self, names: FrozenSet[str]) -> FrozenSet[str]:
        for name in names:
            self.resolve_name(name)
        return names

    def has_name(self, name: str) -> bool:
        return name in self._name_index

    def available_names(self) -> List[str]:
        """All constraint names registered in this region."""
        return list(self._name_index.keys())

    # ── Composition ──

    @staticmethod
    def combine(*regions: FeasibleRegion) -> FeasibleRegion:
        if not regions:
            raise ConstraintError("Must provide at least one region")
        n = regions[0].n_dim
        all_constraints: List[ConvexConstraint] = []
        for r in regions:
            if r.n_dim != n:
                raise ConstraintError(f"Region dimension {r.n_dim} != expected {n}")
            all_constraints.extend(r.constraints)
        return FeasibleRegion(all_constraints, n)

    def add_constraint(self, c: ConvexConstraint) -> FeasibleRegion:
        if c.dimension != self._n:
            raise ConstraintError(f"Constraint dimension {c.dimension} != {self._n}")
        return FeasibleRegion(self._constraints + [c], self._n)

    # ── Linear-specific operations ──

    def with_linear_bound_update(self, row_name: str, new_bound: float) -> FeasibleRegion:
        new_constraints: List[ConvexConstraint] = []
        found = False
        for c in self._constraints:
            if isinstance(c, LinearConstraints) and row_name in c.row_names:
                new_constraints.append(c.with_bound(row_name, new_bound))
                found = True
            else:
                new_constraints.append(c)
        if not found:
            raise ResolutionError(f"Linear row '{row_name}' not found in any constraint block")
        r = FeasibleRegion(new_constraints, self._n)
        r._version = self._version
        return r

    def with_safety_margin(self, margin: float) -> FeasibleRegion:
        if margin >= 1.0:
            return self
        new_constraints: List[ConvexConstraint] = []
        for c in self._constraints:
            if isinstance(c, LinearConstraints):
                new_constraints.append(c.with_safety_margin(margin))
            else:
                new_constraints.append(c)
        r = FeasibleRegion(new_constraints, self._n)
        r._version = self._version
        return r

    # ── Introspection ──

    def has_only_linear(self) -> bool:
        return all(isinstance(c, LinearConstraints) for c in self._constraints)

    def has_linear_row(self, row_name: str) -> bool:
        for c in self._constraints:
            if isinstance(c, LinearConstraints) and row_name in c.row_names:
                return True
        return False

    def assert_linear_row(self, row_name: str) -> None:
        if not self.has_linear_row(row_name):
            raise ResolutionError(
                f"'{row_name}' is not a LinearConstraints row in the active region. "
                "Budgets can only target linear constraint rows."
            )

    def get_linear_matrix(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.has_only_linear():
            return None
        A_blocks = [c.A for c in self._constraints if isinstance(c, LinearConstraints)]
        b_blocks = [c.b for c in self._constraints if isinstance(c, LinearConstraints)]
        return np.vstack(A_blocks), np.concatenate(b_blocks)

    def __repr__(self) -> str:
        types = {}
        for c in self._constraints:
            t = type(c).__name__
            types[t] = types.get(t, 0) + 1
        desc = ", ".join(f"{v}×{k}" for k, v in types.items())
        return f"FeasibleRegion(dim={self._n}, [{desc}], version='{self._version}')"


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTRAINT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def box_constraints(
    lower: ArrayLike, upper: ArrayLike,
    names: Optional[List[str]] = None, tag: str = "",
) -> FeasibleRegion:
    """Box constraints: lower ≤ x ≤ upper. Produces 2n constraints."""
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if lo.shape != hi.shape:
        raise ValidationError("lower and upper must have the same length")
    _validate_finite(lo, "lower")
    _validate_finite(hi, "upper")
    if np.any(lo > hi):
        raise ConstraintError("lower bounds must be ≤ upper bounds element-wise")
    n = lo.shape[0]
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([hi, -lo])
    if names is None:
        names = [f"upper_{i}" for i in range(n)] + [f"lower_{i}" for i in range(n)]
    elif len(names) != 2 * n:
        raise ValidationError(f"Expected {2 * n} names, got {len(names)}")
    lc = LinearConstraints(A, b, names, tags=[tag] * (2 * n))
    return FeasibleRegion([lc], n)


def halfplane(
    weights: ArrayLike, bound: float, name: str = "", tag: str = "",
) -> FeasibleRegion:
    """Single half-plane: weights'x ≤ bound."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    _validate_finite(w, "weights")
    if np.all(w == 0):
        raise ConstraintError("Half-plane weights must be non-zero")
    lc = LinearConstraints(
        w.reshape(1, -1), np.array([float(bound)]),
        names=[name or "halfplane"], tags=[tag],
    )
    return FeasibleRegion([lc], w.shape[0])


def combine_regions(*regions: FeasibleRegion) -> FeasibleRegion:
    """Intersect multiple regions."""
    return FeasibleRegion.combine(*regions)


def ellipsoid(
    Sigma: ArrayLike, center: Optional[ArrayLike] = None,
    radius_sq: float = 1.0, name: str = "ellipsoid", tag: str = "",
) -> QuadraticConstraint:
    """(x − center)'Σ⁻¹(x − center) ≤ radius². Convenience builder."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    try:
        Q = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError as e:
        raise ConstraintError(
            f"Covariance matrix Sigma is singular and cannot be inverted: {e}"
        ) from e
    n = Q.shape[0]
    c = np.zeros(n) if center is None else np.asarray(center, dtype=np.float64).ravel()
    a = -2.0 * Q @ c
    b = radius_sq - float(c @ Q @ c)
    return QuadraticConstraint(Q, a, b, name, tag)


# ═══════════════════════════════════════════════════════════════════════════
#  PROJECTION SOLVER CHAIN
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("numerail")


@dataclass(frozen=True)
class ProjectionResult:
    """Output of the projection solver chain."""
    point: np.ndarray
    iterations: int
    solver_method: str
    converged: bool
    postcheck_passed: bool


def _try_box_clamp(x: np.ndarray, region: FeasibleRegion) -> Optional[np.ndarray]:
    """O(n) closed-form projection for pure-box polytopes."""
    linear = region.get_linear_matrix()
    if linear is None:
        return None
    A, b = linear
    m, n = A.shape
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    for i in range(m):
        nonzero = np.nonzero(A[i])[0]
        if len(nonzero) != 1:
            return None
        j = nonzero[0]
        val = A[i, j]
        if abs(abs(val) - 1.0) > 1e-12:
            return None
        if val > 0:
            upper[j] = min(upper[j], b[i])
        else:
            lower[j] = max(lower[j], -b[i])
    return np.clip(x, lower, upper)


def _try_slsqp(
    x: np.ndarray, region: FeasibleRegion, max_iter: int = 2000,
) -> Tuple[np.ndarray, int]:
    """SLSQP projection: min ½‖y − x‖² s.t. constraints."""
    scipy_constraints = []
    for c in region.constraints:
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda y, _c=c: -_c.evaluate(y),
        })
    result = minimize(
        fun=lambda y: 0.5 * np.dot(y - x, y - x),
        x0=x.copy(),
        jac=lambda y: y - x,
        method="SLSQP",
        constraints=scipy_constraints,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )
    if not result.success:
        raise SolverError(f"SLSQP did not converge: {result.message}")
    if not np.all(np.isfinite(result.x)):
        raise SolverError("SLSQP returned non-finite values")
    return result.x, int(result.nit)


def _try_dykstra(
    x: np.ndarray, region: FeasibleRegion, max_iter: int = 10000, tol: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    """Dykstra's algorithm for projection onto intersection of convex sets."""
    constraints = region.constraints
    k = len(constraints)
    n = region.n_dim
    y = x.copy()
    increments = [np.zeros(n) for _ in range(k)]

    for iteration in range(1, max_iter + 1):
        y_old = y.copy()
        for i, c in enumerate(constraints):
            z = y + increments[i]
            if c.is_satisfied(z, tol=0.0):
                increments[i] = np.zeros(n)
                y = z
            else:
                p = c.project_hint(z)
                if p is None or not c.is_satisfied(p, tol=1e-4):
                    p = _project_single_generic(z, c)
                increments[i] = z - p
                y = p
        if np.linalg.norm(y - y_old) < tol:
            return y, iteration
    return y, max_iter


def _project_single_generic(x: np.ndarray, c: ConvexConstraint) -> np.ndarray:
    """Single-constraint SLSQP projection. Last-resort for Dykstra."""
    result = minimize(
        fun=lambda y: 0.5 * np.dot(y - x, y - x),
        x0=x.copy(),
        jac=lambda y: y - x,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda y, _c=c: -_c.evaluate(y)}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return result.x


def project(
    x: np.ndarray, region: FeasibleRegion,
    max_iter: int = 2000, tol: float = 1e-6,
    dykstra_max_iter: int = 10000,
) -> ProjectionResult:
    """
    Project x onto the feasible region using the fallback solver chain.
    Post-check after each attempt. The solver order depends on constraint types:

    Linear-only regions: box clamp → Dykstra → SLSQP
        Dykstra with half-space projections is fast and reliable for polytopes.
        SLSQP is a fallback only.

    Mixed (nonlinear) regions: box clamp → SLSQP → Dykstra
        SLSQP handles general convex constraints well.
        Dykstra is a fallback for when SLSQP fails to converge.
    """
    if region.is_feasible(x, tol):
        return ProjectionResult(x.copy(), 0, "none", True, True)

    boxed = _try_box_clamp(x, region)
    if boxed is not None and region.is_feasible(boxed, tol):
        return ProjectionResult(boxed, 1, "box_clamp", True, True)

    linear_only = region.get_linear_matrix() is not None

    if linear_only:
        # Dykstra first: cheap half-space projections for polytopes.
        try:
            y_dyk, iters_d = _try_dykstra(x, region, dykstra_max_iter)
            if region.is_feasible(y_dyk, tol):
                return ProjectionResult(y_dyk, iters_d, "dykstra", True, True)
            logger.debug("Linear-only Dykstra output failed post-check, trying SLSQP")
        except Exception as e:
            logger.debug("Linear-only Dykstra raised %s, trying SLSQP", e)

        try:
            y_slsqp, iters_s = _try_slsqp(x, region, max_iter)
            if region.is_feasible(y_slsqp, tol):
                return ProjectionResult(y_slsqp, iters_s, "slsqp", True, True)
            logger.debug("Linear-only SLSQP output failed post-check")
        except Exception as e:
            logger.debug("Linear-only SLSQP raised %s", e)
    else:
        # SLSQP first: general-purpose for nonlinear constraints.
        try:
            y_slsqp, iters_s = _try_slsqp(x, region, max_iter)
            if region.is_feasible(y_slsqp, tol):
                return ProjectionResult(y_slsqp, iters_s, "slsqp", True, True)
            logger.debug("SLSQP output failed post-check, trying Dykstra")
        except Exception as e:
            logger.debug("SLSQP raised %s, trying Dykstra", e)

        try:
            y_dyk, iters_d = _try_dykstra(x, region, dykstra_max_iter)
            if region.is_feasible(y_dyk, tol):
                return ProjectionResult(y_dyk, iters_d, "dykstra", True, True)
            logger.debug("Dykstra output failed post-check")
        except Exception as e:
            logger.debug("Dykstra raised %s", e)

    logger.warning("All projection solvers failed. enforce() will REJECT.")
    return ProjectionResult(x.copy(), 0, "none", False, False)


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def check_feasibility(region: FeasibleRegion) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if the region has a feasible point.

    Pure-linear regions: exact via LP. Returns (True, point) if feasible,
    (False, None) if the LP proves infeasibility.

    Mixed (nonlinear) regions: best-effort projection search over random seeds.
    Returns (True, point) if a feasible point is found. Returns (False, None)
    if no feasible point was found — this does NOT prove the region is empty,
    only that the search did not succeed. Callers must not treat False as
    "provably infeasible" for mixed-constraint regions."""
    linear = region.get_linear_matrix()
    if linear is not None:
        A, b = linear
        res = linprog(np.zeros(region.n_dim), A_ub=A, b_ub=b, method="highs")
        if res.success and res.status == 0 and region.is_feasible(res.x):
            return True, res.x
        if not res.success or res.status != 0:
            return False, None

    seeds = [np.zeros(region.n_dim)]
    rng = np.random.RandomState(0)
    for _ in range(5):
        seeds.append(rng.randn(region.n_dim) * 0.5)

    for seed in seeds:
        if region.is_feasible(seed):
            return True, seed
        proj = project(seed, region)
        if proj.postcheck_passed:
            return True, proj.point

    return False, None


def chebyshev_center(region: FeasibleRegion) -> Tuple[Optional[np.ndarray], float]:
    """Chebyshev center and radius of the largest inscribed ball (linear only)."""
    linear = region.get_linear_matrix()
    if linear is None:
        return None, 0.0
    A, b = linear
    m, n = A.shape
    norms = np.linalg.norm(A, axis=1)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0
    A_ub = np.hstack([A, norms.reshape(-1, 1)])
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return res.x[:n], float(res.x[n])
    return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  ENFORCEMENT OPERATOR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EnforcementConfig:
    """Complete enforcement configuration. Frozen after construction."""
    mode: str = "project"
    max_distance: Optional[float] = None
    dimension_policies: Dict[str, DimensionPolicy] = field(default_factory=dict)
    routing_thresholds: Optional[RoutingThresholds] = None
    hard_wall_constraints: FrozenSet[str] = field(default_factory=frozenset)
    safety_margin: float = 1.0
    solver_max_iter: int = 2000
    solver_tol: float = 1e-6
    dykstra_max_iter: int = 10000

    def __post_init__(self):
        if self.mode not in ("project", "reject", "hybrid"):
            raise ValidationError(f"Invalid mode: '{self.mode}'")
        if self.mode == "hybrid" and self.max_distance is None:
            raise ValidationError("hybrid mode requires max_distance")
        if self.max_distance is not None and self.max_distance <= 0:
            raise ValidationError("max_distance must be positive")
        if self.safety_margin <= 0 or self.safety_margin > 1.0:
            raise ValidationError("safety_margin must be in (0, 1.0]")
        if self.solver_max_iter < 1:
            raise ValidationError("solver_max_iter must be >= 1")
        if self.solver_tol <= 0:
            raise ValidationError("solver_tol must be > 0")
        if self.dykstra_max_iter < 1:
            raise ValidationError("dykstra_max_iter must be >= 1")


@dataclass(frozen=True)
class RollbackResult:
    """Normalized rollback return type across all API surfaces."""
    rolled_back: bool
    audit_hash: Optional[str] = None

    def __bool__(self) -> bool:
        return self.rolled_back


@dataclass
class EnforcementOutput:
    """Complete, auditable record of a single enforcement decision."""
    result: EnforcementResult
    original_vector: np.ndarray
    enforced_vector: np.ndarray
    distance: float
    violated_constraints: Tuple[str, ...]
    violation_magnitudes: Dict[str, float]
    binding_constraints: Tuple[str, ...]
    solver_method: str
    iterations: int
    timestamp: str
    region_version: str = ""
    routing_decision: Optional[RoutingDecision] = None
    flagged_dimensions: Tuple[str, ...] = ()
    prev_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "result": self.result.value,
            "original_vector": [round(float(v), 15) for v in self.original_vector],
            "enforced_vector": [round(float(v), 15) for v in self.enforced_vector],
            "distance": round(self.distance, 15),
            "violated_constraints": list(self.violated_constraints),
            "violation_magnitudes": {k: round(v, 10) for k, v in self.violation_magnitudes.items()},
            "binding_constraints": list(self.binding_constraints),
            "solver_method": self.solver_method,
            "iterations": self.iterations,
            "timestamp": self.timestamp,
            "region_version": self.region_version,
            "prev_hash": self.prev_hash,
        }
        if self.routing_decision is not None:
            d["routing_decision"] = self.routing_decision.value
        if self.flagged_dimensions:
            d["flagged_dimensions"] = list(self.flagged_dimensions)
        return d

    def hash(self) -> str:
        return hashlib.sha256(_deterministic_json(self.to_dict()).encode()).hexdigest()


def enforce(
    vector: ArrayLike,
    region: FeasibleRegion,
    config: Optional[EnforcementConfig] = None,
    schema: Optional["Schema"] = None,
    prev_hash: str = "",
) -> EnforcementOutput:
    """
    Enforce geometric constraints on a vector.

    GUARANTEE (Theorem 1):
        If result ∈ {APPROVE, PROJECT}, enforced_vector ∈ F(region).
        For ALL inputs. For ALL solvers.
    """
    cfg = config or EnforcementConfig()
    x = _validate_vector(vector, region.n_dim, "vector")
    ts = _utc_now()

    # Fail fast on misconfigured dimension policies when schema is available.
    if schema is not None and cfg.dimension_policies:
        schema.validate_enforcement_config(cfg)

    effective = region.with_safety_margin(cfg.safety_margin) if cfg.safety_margin < 1.0 else region

    def _dim_name_to_idx(name: str) -> Optional[int]:
        if schema is not None:
            try:
                return schema.field_index(name)
            except SchemaError:
                return None
        return None

    def _dim_idx_to_name(idx: int) -> str:
        if schema is not None and idx < len(schema.fields):
            return schema.fields[idx]
        return str(idx)

    def _out(
        result: EnforcementResult,
        enforced: np.ndarray,
        dist: float = 0.0,
        violated: Tuple[str, ...] = (),
        magnitudes: Optional[Dict[str, float]] = None,
        binding: Tuple[str, ...] = (),
        solver: str = "none",
        iters: int = 0,
        routing: Optional[RoutingDecision] = None,
        flagged: Tuple[str, ...] = (),
    ) -> EnforcementOutput:
        # Defense-in-depth: the Numerail invariant made explicit.
        # Control flow already guarantees this; this check catches implementation bugs.
        # Uses explicit raise (not assert) so it cannot be stripped by python -O.
        if result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
            if not effective.is_feasible(enforced, cfg.solver_tol):
                raise AssertionError(
                    "Numerail invariant violated: emitted vector must satisfy "
                    "the exact combined checker for the active feasible region. "
                    "This check should never fire in correctly functioning code."
                )
        return EnforcementOutput(
            result=result, original_vector=x.copy(), enforced_vector=enforced.copy(),
            distance=dist, violated_constraints=violated,
            violation_magnitudes=magnitudes or {}, binding_constraints=binding,
            solver_method=solver, iterations=iters, timestamp=ts,
            region_version=effective.version, routing_decision=routing,
            flagged_dimensions=flagged, prev_hash=prev_hash,
        )

    # STEP 1: POST-CHECK INPUT → APPROVE if feasible
    if effective.is_feasible(x):
        return _out(EnforcementResult.APPROVE, x, binding=effective.binding_names(x))

    viol_dict = effective.violations(x)
    violated_names = tuple(viol_dict.keys())

    # STEP 2: Hard wall check
    if cfg.hard_wall_constraints:
        try:
            effective.resolve_names(cfg.hard_wall_constraints)
        except ResolutionError as e:
            raise ResolutionError(f"Hard wall resolution failed: {e}")
        if cfg.hard_wall_constraints & frozenset(violated_names):
            return _out(
                EnforcementResult.REJECT, x, violated=violated_names,
                magnitudes=viol_dict, routing=RoutingDecision.HARD_REJECT,
            )

    # STEP 3: REJECT mode gate
    if cfg.mode == "reject":
        return _out(EnforcementResult.REJECT, x, violated=violated_names, magnitudes=viol_dict)

    # STEP 4: PROJECT — call the fallback solver chain
    proj = project(
        x, effective,
        max_iter=cfg.solver_max_iter,
        tol=cfg.solver_tol,
        dykstra_max_iter=cfg.dykstra_max_iter,
    )
    distance = float(np.linalg.norm(proj.point - x))

    # STEP 5: POST-CHECK SOLVER OUTPUT
    if not proj.postcheck_passed:
        logger.warning("All solvers failed containment post-check. Failing closed to REJECT.")
        return _out(
            EnforcementResult.REJECT, x, dist=distance,
            violated=violated_names, magnitudes=viol_dict,
            solver=proj.solver_method, iters=proj.iterations,
        )

    binding = effective.binding_names(proj.point)

    # STEPS 6–8: Operational policy (can only ADD reject conditions)

    flagged_dims: List[str] = []
    for dim_name, policy in cfg.dimension_policies.items():
        dim_idx = _dim_name_to_idx(dim_name)
        if dim_idx is None or dim_idx >= len(x):
            logger.warning(
                "dimension_policies key '%s' could not be resolved against schema. "
                "This should have been caught by validate_enforcement_config().",
                dim_name,
            )
            continue
        if abs(proj.point[dim_idx] - x[dim_idx]) > 1e-9:
            if policy == DimensionPolicy.PROJECTION_FORBIDDEN:
                return _out(
                    EnforcementResult.REJECT, x, dist=distance,
                    violated=violated_names, magnitudes=viol_dict,
                    binding=binding, solver=proj.solver_method, iters=proj.iterations,
                    routing=RoutingDecision.HARD_REJECT, flagged=(_dim_idx_to_name(dim_idx),),
                )
            elif policy == DimensionPolicy.PROJECT_WITH_FLAG:
                flagged_dims.append(_dim_idx_to_name(dim_idx))

    routing: Optional[RoutingDecision] = None
    if cfg.routing_thresholds is not None:
        routing = _compute_routing(distance, cfg.routing_thresholds)
        if routing == RoutingDecision.HARD_REJECT:
            return _out(
                EnforcementResult.REJECT, x, dist=distance,
                violated=violated_names, magnitudes=viol_dict,
                binding=binding, solver=proj.solver_method, iters=proj.iterations,
                routing=routing,
            )

    if cfg.mode == "hybrid" and cfg.max_distance is not None:
        if distance > cfg.max_distance:
            return _out(
                EnforcementResult.REJECT, x, dist=distance,
                violated=violated_names, magnitudes=viol_dict,
                binding=binding, solver=proj.solver_method, iters=proj.iterations,
            )

    return _out(
        EnforcementResult.PROJECT, proj.point, dist=distance,
        violated=violated_names, magnitudes=viol_dict,
        binding=binding, solver=proj.solver_method, iters=proj.iterations,
        routing=routing, flagged=tuple(flagged_dims),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NormalizerRange:
    """Affine normalization: raw ↔ [0,1] mapping."""
    lo: float
    hi: float

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValidationError(f"NormalizerRange: hi ({self.hi}) must be > lo ({self.lo})")

    def normalize(self, val: float) -> float:
        return (val - self.lo) / (self.hi - self.lo)

    def denormalize(self, val: float) -> float:
        return val * (self.hi - self.lo) + self.lo


class Schema:
    """Vectorization/devectorization with affine normalization."""

    def __init__(
        self,
        fields: Sequence[str],
        normalizers: Optional[Dict[str, Tuple[float, float]]] = None,
        defaults: Optional[Dict[str, float]] = None,
    ):
        if not fields:
            raise SchemaError("Fields list must be non-empty")
        if len(fields) != len(set(fields)):
            raise SchemaError(f"Duplicate field names: {list(fields)}")
        self._fields: Tuple[str, ...] = tuple(fields)
        self._field_index: Dict[str, int] = {f: i for i, f in enumerate(fields)}
        self._normalizers: Dict[str, NormalizerRange] = {}
        if normalizers:
            for fname, (lo, hi) in normalizers.items():
                if fname not in self._field_index:
                    raise SchemaError(f"Normalizer for unknown field '{fname}'")
                if not np.isfinite(lo) or not np.isfinite(hi):
                    raise SchemaError(
                        f"Normalizer for '{fname}' must have finite bounds"
                    )
                if hi <= lo:
                    raise SchemaError(
                        f"Normalizer for '{fname}' must have hi > lo, "
                        f"got lo={lo}, hi={hi}"
                    )
                self._normalizers[fname] = NormalizerRange(lo, hi)
        self._defaults: Dict[str, float] = dict(defaults) if defaults else {}

    @property
    def fields(self) -> Tuple[str, ...]:
        return self._fields

    @property
    def dimension(self) -> int:
        return len(self._fields)

    def field_index(self, name: str) -> int:
        if name not in self._field_index:
            raise SchemaError(f"Unknown field '{name}'")
        return self._field_index[name]

    def vectorize(self, values: Dict[str, Any]) -> np.ndarray:
        """Dict → normalized vector."""
        vec = np.empty(len(self._fields), dtype=np.float64)
        for i, f in enumerate(self._fields):
            if f in values:
                val = float(values[f])
            elif f in self._defaults:
                val = self._defaults[f]
            else:
                raise SchemaError(f"Field '{f}' missing from values and has no default")
            if f in self._normalizers:
                val = self._normalizers[f].normalize(val)
            if not np.isfinite(val):
                raise ValidationError(f"Field '{f}' produced non-finite value: {val}")
            vec[i] = val
        return vec

    def devectorize(self, vec: np.ndarray) -> Dict[str, float]:
        """Normalized vector → dict."""
        if len(vec) != len(self._fields):
            raise SchemaError(f"Vector length {len(vec)} != schema dimension {len(self._fields)}")
        result: Dict[str, float] = {}
        for i, f in enumerate(self._fields):
            val = float(vec[i])
            if f in self._normalizers:
                val = self._normalizers[f].denormalize(val)
            result[f] = val
        return result

    def validate_region(self, region: FeasibleRegion) -> None:
        if region.n_dim != self.dimension:
            raise ValidationError(
                f"Region has {region.n_dim} dimensions, schema has {self.dimension} fields"
            )

    def denormalize_field(self, field_name: str, normalized_value: float) -> float:
        if field_name in self._normalizers:
            return self._normalizers[field_name].denormalize(normalized_value)
        return normalized_value

    def normalize_field(self, field_name: str, raw_value: float) -> float:
        if field_name in self._normalizers:
            return self._normalizers[field_name].normalize(raw_value)
        return raw_value

    def has_normalizer(self, field_name: str) -> bool:
        return field_name in self._normalizers

    def validate_enforcement_config(self, config: "EnforcementConfig") -> None:
        """Validate that config references only known schema fields.
        Catches dimension_policies typos at construction time, not enforcement time."""
        for dim_name in config.dimension_policies:
            if dim_name not in self._field_index:
                raise SchemaError(
                    f"dimension_policies references unknown field '{dim_name}'. "
                    f"Known fields: {list(self._fields)}"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  BUDGET TRACKER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BudgetSpec:
    """Budget specification. All references are by name, not index.

    weight_map overrides dimension_name + weight when provided.
    weight_map maps field names to weights: consumption = sum(w * enforced[f] for f, w in weight_map).
    When weight_map is None, it is derived as {dimension_name: weight} for backward compatibility.
    """
    name: str
    constraint_name: str
    dimension_name: str = ""
    weight: float = 1.0
    initial: float = 0.0
    consumption_mode: str = "nonnegative"
    weight_map: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if not self.name:
            raise ValidationError("BudgetSpec.name must be non-empty")
        if not self.constraint_name:
            raise ValidationError("BudgetSpec.constraint_name must be non-empty")
        if self.weight_map is None and not self.dimension_name:
            raise ValidationError("BudgetSpec requires dimension_name or weight_map")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValidationError(
                f"BudgetSpec.weight must be finite and >= 0, got {self.weight}"
            )
        if not np.isfinite(self.initial) or self.initial < 0:
            raise ValidationError(
                f"BudgetSpec.initial must be finite and >= 0, got {self.initial}"
            )
        if self.consumption_mode not in ("nonnegative", "abs", "raw"):
            raise ValidationError(
                "BudgetSpec.consumption_mode must be one of "
                "('nonnegative', 'abs', 'raw')"
            )

    @property
    def effective_weight_map(self) -> Dict[str, float]:
        """Resolved weight map. Falls back to {dimension_name: weight}."""
        if self.weight_map is not None:
            return dict(self.weight_map)
        return {self.dimension_name: self.weight}


class BudgetTracker:
    """Name-based budget tracker with rollback support."""

    def __init__(self, max_rollback_history: int = 10000):
        self._specs: Dict[str, BudgetSpec] = {}
        self._consumed: Dict[str, float] = {}
        self._action_deltas: Dict[str, Dict[str, float]] = {}
        self._action_order: deque = deque()
        self._lock = threading.Lock()
        self._max_rollback = max_rollback_history

    def register(self, spec: BudgetSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec
            self._consumed[spec.name] = 0.0

    @property
    def has_budgets(self) -> bool:
        return len(self._specs) > 0

    def get_bound_updates(
        self, region: FeasibleRegion, schema: Schema,
    ) -> Dict[str, float]:
        with self._lock:
            updates: Dict[str, float] = {}
            for budget_name, spec in self._specs.items():
                region.assert_linear_row(spec.constraint_name)
                remaining = spec.initial - self._consumed[budget_name]
                updates[spec.constraint_name] = remaining
            return updates

    def record_consumption(
        self, enforced_vector: np.ndarray, action_id: str, schema: Schema,
    ) -> None:
        with self._lock:
            deltas: Dict[str, float] = {}
            for budget_name, spec in self._specs.items():
                wm = spec.effective_weight_map
                raw = 0.0
                for field_name, w in wm.items():
                    dim_idx = schema.field_index(field_name)
                    if dim_idx < len(enforced_vector):
                        raw += float(enforced_vector[dim_idx]) * w
                if spec.consumption_mode == "nonnegative":
                    delta = max(0.0, raw)
                elif spec.consumption_mode == "abs":
                    delta = abs(raw)
                else:
                    delta = raw
                self._consumed[budget_name] += delta
                deltas[budget_name] = delta
            self._action_deltas[action_id] = deltas
            self._action_order.append(action_id)
            while (self._max_rollback > 0
                   and len(self._action_deltas) > self._max_rollback):
                oldest = self._action_order.popleft()
                self._action_deltas.pop(oldest, None)

    def rollback(self, action_id: str) -> bool:
        with self._lock:
            deltas = self._action_deltas.pop(action_id, None)
            if deltas is None:
                return False
            for budget_name, delta in deltas.items():
                self._consumed[budget_name] = max(0.0, self._consumed[budget_name] - delta)
            return True

    def status(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                name: {
                    "initial": spec.initial,
                    "consumed": round(self._consumed[name], 6),
                    "remaining": round(spec.initial - self._consumed[name], 6),
                }
                for name, spec in self._specs.items()
            }

    def set_consumed(self, budget_name: str, consumed: float) -> None:
        """Set the consumed total for a budget. Thread-safe.

        Used by the service layer to synchronize engine state with
        the persisted budget remaining from the runtime repository.
        """
        with self._lock:
            if budget_name not in self._specs:
                raise ValidationError(
                    f"Unknown budget '{budget_name}' in set_consumed()"
                )
            self._consumed[budget_name] = max(0.0, consumed)


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIT CHAIN
# ═══════════════════════════════════════════════════════════════════════════

class AuditChain:
    """Append-only, hash-linked enforcement log.
    Supports max_records for bounded memory. When exceeded, oldest
    records are evicted. The chain validates from the retention boundary forward."""

    def __init__(self, max_records: int = 0):
        """max_records=0 means unlimited (backward compatible)."""
        self._records: List[Dict[str, Any]] = []
        self._hashes: List[str] = []
        self._last_hash: str = ""
        self._lock = threading.Lock()
        self._max_records = max_records
        self._evicted_count: int = 0

    def append(self, output: EnforcementOutput) -> str:
        with self._lock:
            record = output.to_dict()
            record["prev_hash"] = self._last_hash
            h = hashlib.sha256(_deterministic_json(record).encode()).hexdigest()
            record["hash"] = h
            self._records.append(record)
            self._hashes.append(h)
            self._last_hash = h
            if self._max_records > 0 and len(self._records) > self._max_records:
                evict = len(self._records) - self._max_records
                self._records = self._records[evict:]
                self._hashes = self._hashes[evict:]
                self._evicted_count += evict
            return h

    def verify(self) -> Tuple[bool, int]:
        """Walk the chain, recompute every hash. Returns (valid, depth).
        After eviction, uses the first retained record's prev_hash as anchor."""
        with self._lock:
            records = [r.copy() for r in self._records]
        prev = records[0].get("prev_hash", "") if records else ""
        for i, rec in enumerate(records):
            if rec.get("prev_hash") != prev:
                return False, i
            check = {k: v for k, v in rec.items() if k != "hash"}
            h = hashlib.sha256(_deterministic_json(check).encode()).hexdigest()
            if h != rec["hash"]:
                return False, i
            prev = h
        return True, len(records)

    @property
    def last_hash(self) -> str:
        with self._lock:
            return self._last_hash

    @property
    def length(self) -> int:
        """Total records appended (including evicted)."""
        with self._lock:
            return len(self._records) + self._evicted_count

    @property
    def retained_length(self) -> int:
        with self._lock:
            return len(self._records)

    def export(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.copy() for r in self._records]


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Thread-safe operational metrics collection."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {"approve": 0, "project": 0, "reject": 0}
        self._distances: List[float] = []
        self._violation_freq: Dict[str, int] = {}
        self._binding_freq: Dict[str, int] = {}
        self._solver_freq: Dict[str, int] = {}
        self._iter_counts: List[int] = []

    def record(self, output: EnforcementOutput) -> None:
        with self._lock:
            self._counts[output.result.value] += 1
            if output.distance > 0:
                self._distances.append(output.distance)
            self._iter_counts.append(output.iterations)
            for name in output.violated_constraints:
                self._violation_freq[name] = self._violation_freq.get(name, 0) + 1
            for name in output.binding_constraints:
                self._binding_freq[name] = self._binding_freq.get(name, 0) + 1
            self._solver_freq[output.solver_method] = (
                self._solver_freq.get(output.solver_method, 0) + 1
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(self._counts.values())
            if total == 0:
                return {"total": 0}
            dists = self._distances or [0.0]
            return {
                "total": total,
                "approve_rate": round(self._counts["approve"] / total, 4),
                "project_rate": round(self._counts["project"] / total, 4),
                "reject_rate": round(self._counts["reject"] / total, 4),
                "distance_mean": round(float(np.mean(dists)), 6),
                "distance_p99": round(
                    float(np.percentile(dists, 99)) if len(dists) > 1 else dists[0], 6
                ),
                "distance_max": round(float(np.max(dists)), 6),
                "solver_distribution": dict(self._solver_freq),
                "top_violations": sorted(
                    self._violation_freq.items(), key=lambda kv: -kv[1]
                )[:10],
                "top_binding": sorted(
                    self._binding_freq.items(), key=lambda kv: -kv[1]
                )[:10],
            }

    def reset(self) -> None:
        with self._lock:
            self._counts = {"approve": 0, "project": 0, "reject": 0}
            self._distances.clear()
            self._violation_freq.clear()
            self._binding_freq.clear()
            self._solver_freq.clear()
            self._iter_counts.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  REGION VERSION STORE
# ═══════════════════════════════════════════════════════════════════════════

class RegionVersionStore:
    """Versioned, immutable region history. Thread-safe."""

    def __init__(self, initial: FeasibleRegion, max_versions: int = 100):
        self._counter = 0
        self._counter_lock = threading.Lock()
        vid = self._make_id()
        self._current = initial.with_version(vid)
        self._history: List[FeasibleRegion] = [self._current]
        self._lock = threading.Lock()
        self._max_versions = max_versions

    def _make_id(self) -> str:
        with self._counter_lock:
            self._counter += 1
            seq = self._counter
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3]
        return f"{ts}-{seq:06d}Z"

    def _trim_history(self) -> None:
        if self._max_versions > 0 and len(self._history) > self._max_versions:
            self._history = self._history[-(self._max_versions):]

    @property
    def current(self) -> FeasibleRegion:
        with self._lock:
            return self._current

    def update(self, new_region: FeasibleRegion) -> FeasibleRegion:
        with self._lock:
            vid = self._make_id()
            versioned = new_region.with_version(vid)
            self._current = versioned
            self._history.append(versioned)
            self._trim_history()
            return versioned

    def rollback_to(self, version: str) -> Optional[FeasibleRegion]:
        with self._lock:
            for r in self._history:
                if r.version == version:
                    vid = self._make_id()
                    restored = r.with_version(vid)
                    self._current = restored
                    self._history.append(restored)
                    self._trim_history()
                    return restored
            return None

    @property
    def history_length(self) -> int:
        return len(self._history)


# ═══════════════════════════════════════════════════════════════════════════
#  FEEDBACK SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_feedback(
    output: EnforcementOutput,
    region: FeasibleRegion,
    schema: Schema,
    budget_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Translate enforcement output into agent-facing structured feedback."""
    fb: Dict[str, Any] = {
        "result": output.result.value,
        "modified": output.result == EnforcementResult.PROJECT,
    }
    if output.result == EnforcementResult.APPROVE:
        fb["message"] = "Action approved. All constraints satisfied."
        if budget_status:
            fb["budget"] = budget_status
        return fb

    if output.violated_constraints:
        fb["violations"] = [
            {"constraint": name, "magnitude": round(output.violation_magnitudes.get(name, 0), 6)}
            for name in output.violated_constraints
        ]

    if output.result == EnforcementResult.PROJECT:
        changes: Dict[str, Dict[str, float]] = {}
        for i, f in enumerate(schema.fields):
            orig = float(output.original_vector[i])
            enf = float(output.enforced_vector[i])
            if abs(orig - enf) > 1e-9:
                orig = schema.denormalize_field(f, orig)
                enf = schema.denormalize_field(f, enf)
                changes[f] = {"proposed": round(orig, 6), "enforced": round(enf, 6)}
        fb["corrections"] = changes
        fb["distance"] = round(output.distance, 6)
        binding_str = ", ".join(output.binding_constraints) if output.binding_constraints else "none"
        fb["binding"] = list(output.binding_constraints)
        fb["message"] = f"Action corrected (distance {output.distance:.4f}). Binding: {binding_str}."
    else:
        fb["message"] = "Action rejected."

    if output.routing_decision:
        fb["routing"] = output.routing_decision.value
    if budget_status:
        fb["budget"] = budget_status
    return fb


# ═══════════════════════════════════════════════════════════════════════════
#  NUMERAIL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class NumerailSystem:
    """
    Complete Numerail enforcement system. Production-ready integration of
    enforcement, region lifecycle, budget tracking, audit, and metrics.
    """

    @dataclass
    class Result:
        """The return type of enforce()."""
        enforced_values: Dict[str, float]
        output: EnforcementOutput
        feedback: Dict[str, Any]
        audit_hash: str

    def __init__(
        self,
        schema: Schema,
        region_or_polytope: Union[FeasibleRegion, Any],
        config: Optional[EnforcementConfig] = None,
    ):
        # Accept FeasibleRegion or Polytope (backward compat)
        if isinstance(region_or_polytope, FeasibleRegion):
            region = region_or_polytope
        else:
            # Try Polytope.as_region() for backward compat
            try:
                region = region_or_polytope.as_region()
            except AttributeError:
                raise ValidationError(
                    f"Expected FeasibleRegion or Polytope, got {type(region_or_polytope)}"
                )
        schema.validate_region(region)
        self._schema = schema
        self._config = config or EnforcementConfig()
        # CHANGE 1+2: Fail fast on misconfigured policies at construction time.
        if self._config.dimension_policies:
            self._schema.validate_enforcement_config(self._config)
        if self._config.hard_wall_constraints:
            for hw_name in self._config.hard_wall_constraints:
                if not region.has_name(hw_name):
                    raise ResolutionError(
                        f"hard_wall_constraints references unknown constraint "
                        f"'{hw_name}' in the initial region. "
                        f"Available: {region.available_names()}"
                    )
        self._versions = RegionVersionStore(region)
        self._budgets = BudgetTracker()
        self._audit = AuditChain()
        self._metrics = MetricsCollector()
        self._enforce_lock = threading.Lock()
        self._trusted_fields: FrozenSet[str] = frozenset()

    def set_trusted_fields(self, fields: FrozenSet[str]) -> None:
        """Declare which fields are server-authoritative (trusted context)."""
        for f in fields:
            try:
                self._schema.field_index(f)
            except SchemaError:
                raise SchemaError(
                    f"trusted_fields references unknown field '{f}'. "
                    f"Known fields: {list(self._schema.fields)}"
                )
        self._trusted_fields = frozenset(fields)

    # ── Primary API ──

    def enforce(
        self,
        values: Dict[str, Any],
        action_id: Optional[str] = None,
        trusted_context: Optional[Dict[str, float]] = None,
    ) -> Result:
        """Enforce constraints on structured values. Thread-safe, atomic.

        If trusted_context is provided and trusted_fields have been declared,
        server-authoritative values overwrite the corresponding fields in
        the proposed values before enforcement. Both the raw proposal and
        the merged values are recorded in the feedback for auditability.
        """
        with self._enforce_lock:
            base = self._versions.current

            # Apply trusted context merge if configured
            raw_values = dict(values)
            if trusted_context and self._trusted_fields:
                merged_values = merge_trusted_context(
                    raw_values, trusted_context, self._trusted_fields,
                )
            else:
                merged_values = raw_values

            if self._budgets.has_budgets:
                updates = self._budgets.get_bound_updates(base, self._schema)
                effective = base
                for row_name, new_bound in updates.items():
                    effective = effective.with_linear_bound_update(row_name, new_bound)
                digest = hashlib.sha256(
                    _deterministic_json({"budget_bounds": updates}).encode()
                ).hexdigest()[:12]
                effective = effective.with_version(f"{base.version}|bud:{digest}")
            else:
                effective = base

            vec = self._schema.vectorize(merged_values)
            output = enforce(
                vec, effective, self._config,
                schema=self._schema, prev_hash=self._audit.last_hash,
            )

            audit_hash = self._audit.append(output)
            self._metrics.record(output)

            aid = action_id or f"a_{self._audit.length}"
            if output.result in (EnforcementResult.APPROVE, EnforcementResult.PROJECT):
                if self._budgets.has_budgets:
                    self._budgets.record_consumption(output.enforced_vector, aid, self._schema)

            enforced_values = self._schema.devectorize(output.enforced_vector)

            fb_budget = self._budgets.status() if self._budgets.has_budgets else None
            feedback = synthesize_feedback(output, effective, self._schema, fb_budget)
            # Record both raw and merged for trusted-context auditability
            if trusted_context and self._trusted_fields:
                feedback["raw_values"] = raw_values
                feedback["merged_values"] = merged_values
                feedback["trusted_fields_applied"] = sorted(self._trusted_fields)

            return NumerailSystem.Result(
                enforced_values=enforced_values, output=output,
                feedback=feedback, audit_hash=audit_hash,
            )

    def rollback(self, action_id: str) -> RollbackResult:
        """Rollback budget consumption and sync region."""
        with self._enforce_lock:
            success = self._budgets.rollback(action_id)
            if success and self._budgets.has_budgets:
                base = self._versions.current
                updates = self._budgets.get_bound_updates(base, self._schema)
                synced = base
                for row_name, new_bound in updates.items():
                    synced = synced.with_linear_bound_update(row_name, new_bound)
                self._versions.update(synced)
            return RollbackResult(rolled_back=success)

    # ── Budget management ──

    def register_budget(self, spec: BudgetSpec) -> None:
        """Register a budget. Validates names at registration time."""
        # Validate field references
        wm = spec.effective_weight_map
        for field_name in wm:
            try:
                self._schema.field_index(field_name)
            except SchemaError:
                raise SchemaError(
                    f"Budget '{spec.name}': weight references unknown field "
                    f"'{field_name}' (schema fields: {list(self._schema.fields)})"
                )
        region = self._versions.current
        try:
            region.assert_linear_row(spec.constraint_name)
        except ResolutionError:
            raise ResolutionError(
                f"Budget '{spec.name}': constraint_name '{spec.constraint_name}' "
                f"is not a linear constraint row in the current region"
            )
        self._budgets.register(spec)

    def budget_status(self) -> Dict[str, Dict[str, float]]:
        return self._budgets.status()

    @property
    def has_budgets(self) -> bool:
        """Whether any budgets are registered."""
        return self._budgets.has_budgets

    def sync_budget_consumed(self, budget_name: str, consumed: float) -> None:
        """Set the consumed total for a budget from external state.

        Used by the service layer to synchronize engine state with the
        persisted budget remaining from the runtime repository.
        """
        self._budgets.set_consumed(budget_name, consumed)

    # ── Region management ──

    def add_constraints(self, new: Union[FeasibleRegion, Any]) -> FeasibleRegion:
        current = self._versions.current
        if not isinstance(new, FeasibleRegion):
            try:
                new = new.as_region()
            except AttributeError:
                raise ValidationError(f"Expected FeasibleRegion, got {type(new)}")
        combined = FeasibleRegion.combine(current, new)
        return self._versions.update(combined)

    def replace_region(self, new: Union[FeasibleRegion, Any]) -> FeasibleRegion:
        if not isinstance(new, FeasibleRegion):
            try:
                new = new.as_region()
            except AttributeError:
                raise ValidationError(f"Expected FeasibleRegion, got {type(new)}")
        self._schema.validate_region(new)
        return self._versions.update(new)

    def rollback_region(self, version: str) -> Optional[FeasibleRegion]:
        return self._versions.rollback_to(version)

    @property
    def region(self) -> FeasibleRegion:
        return self._versions.current

    # ── Analysis ──

    def check_feasibility(self) -> Tuple[bool, Optional[np.ndarray]]:
        return check_feasibility(self._versions.current)

    def chebyshev_radius(self) -> float:
        _, r = chebyshev_center(self._versions.current)
        return r

    # ── Audit & Metrics ──

    def verify_audit(self) -> Tuple[bool, int]:
        return self._audit.verify()

    def export_audit(self) -> List[Dict[str, Any]]:
        return self._audit.export()

    @property
    def audit_length(self) -> int:
        return self._audit.length

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.summary()

    def reset_metrics(self) -> None:
        self._metrics.reset()

    # ── Configuration-driven factory ──

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> NumerailSystem:
        """Construct from a JSON-compatible dict."""
        sc = config_dict.get("action_schema", config_dict.get("schema", {}))
        schema = Schema(
            fields=sc["fields"],
            normalizers={k: tuple(v) for k, v in sc.get("normalizers", {}).items()},
            defaults=sc.get("defaults", {}),
        )

        constraints: List[ConvexConstraint] = []
        n_dim = len(sc["fields"])

        if "polytope" in config_dict:
            pd = config_dict["polytope"]
            constraints.append(LinearConstraints(
                pd["A"], pd["b"],
                names=pd.get("names"),
                tags=pd.get("tags") or pd.get("authorities"),
            ))

        for qc in config_dict.get("quadratic_constraints", []):
            constraints.append(QuadraticConstraint(
                qc["Q"], qc["a"], qc["b"],
                constraint_name=qc.get("name", "quadratic"),
                constraint_tag=qc.get("tag") or qc.get("authority", ""),
            ))

        for sc_def in config_dict.get("socp_constraints", []):
            constraints.append(SOCPConstraint(
                sc_def["M"], sc_def["q"], sc_def["c"], sc_def["d"],
                constraint_name=sc_def.get("name", "socp"),
                constraint_tag=sc_def.get("tag") or sc_def.get("authority", ""),
            ))

        for pc in config_dict.get("psd_constraints", []):
            constraints.append(PSDConstraint(
                pc["A0"], pc["A_list"],
                constraint_name=pc.get("name", "psd"),
                constraint_tag=pc.get("tag") or pc.get("authority", ""),
            ))

        if not constraints:
            raise ValueError("Config must define at least one constraint block")

        region = FeasibleRegion(constraints, n_dim)

        ec = config_dict.get("enforcement", {})
        dim_policies = {k: DimensionPolicy(v) for k, v in ec.get("dimension_policies", {}).items()}
        rt = ec.get("routing_thresholds")
        routing = RoutingThresholds(**rt) if rt else None

        config = EnforcementConfig(
            mode=ec.get("mode", "project"),
            max_distance=ec.get("max_distance"),
            dimension_policies=dim_policies,
            routing_thresholds=routing,
            hard_wall_constraints=frozenset(ec.get("hard_wall_constraints", [])),
            safety_margin=ec.get("safety_margin", 1.0),
            solver_max_iter=ec.get("solver_max_iter", 2000),
            solver_tol=ec.get("solver_tol", 1e-6),
            dykstra_max_iter=ec.get("dykstra_max_iter", 10000),
        )

        system = cls(schema, region, config)

        # Parse trusted fields if present
        trusted_fields = config_dict.get("trusted_fields", [])
        if trusted_fields:
            system.set_trusted_fields(frozenset(trusted_fields))

        for b in config_dict.get("budgets", []):
            # Support both canonical weight map and scalar shorthand
            weight_map_raw = b.get("weight")
            if isinstance(weight_map_raw, dict):
                # Canonical: {"amount": 1.0, "risk": 0.5}
                system.register_budget(BudgetSpec(
                    name=b["name"],
                    constraint_name=b["constraint_name"],
                    initial=b["initial"],
                    consumption_mode=b.get("consumption_mode", b.get("mode", "nonnegative")),
                    weight_map=weight_map_raw,
                ))
            else:
                # Shorthand: dimension_name + scalar weight
                system.register_budget(BudgetSpec(
                    name=b["name"],
                    constraint_name=b["constraint_name"],
                    dimension_name=b.get("dimension_name", ""),
                    weight=float(weight_map_raw) if weight_map_raw is not None else 1.0,
                    initial=b["initial"],
                    consumption_mode=b.get("consumption_mode", b.get("mode", "nonnegative")),
                ))

        return system


# ═══════════════════════════════════════════════════════════════════════════
#  TRUSTED CONTEXT
# ═══════════════════════════════════════════════════════════════════════════


def merge_trusted_context(
    raw_values: Dict[str, Any],
    trusted_context: Dict[str, float],
    trusted_fields: FrozenSet[str],
) -> Dict[str, float]:
    """Merge server-authoritative trusted context into a raw action dict.

    Fields listed in trusted_fields are overwritten with values from
    trusted_context. All other fields pass through from raw_values.
    This prevents the agent from controlling fields that determine
    its own admissibility (e.g., fraud scores, system-measured quotas).
    """
    merged: Dict[str, float] = {k: float(v) for k, v in raw_values.items()}
    for field in trusted_fields:
        if field in trusted_context:
            merged[field] = float(trusted_context[field])
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

class Polytope:
    """
    Convex polytope: {x ∈ ℝⁿ : Ax ≤ b}. Immutable after construction.
    Backward-compatible wrapper around LinearConstraints.
    """

    __slots__ = ("_lc", "_version")

    def __init__(
        self,
        A: ArrayLike,
        b: ArrayLike,
        names: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        version: str = "",
        # Backward compat alias
        authorities: Optional[Sequence[str]] = None,
    ):
        effective_tags = tags or authorities
        self._lc = LinearConstraints(A, b, names, effective_tags, constraint_name="linear")
        self._version = version

    @property
    def A(self) -> np.ndarray:
        return self._lc.A

    @property
    def b(self) -> np.ndarray:
        return self._lc.b

    @property
    def names(self) -> Tuple[str, ...]:
        return self._lc.row_names

    @property
    def tags(self) -> Tuple[str, ...]:
        return self._lc.row_tags

    @property
    def authorities(self) -> Tuple[str, ...]:
        """Backward compat alias for tags."""
        return self._lc.row_tags

    @property
    def version(self) -> str:
        return self._version

    @property
    def n_constraints(self) -> int:
        return self._lc.n_rows

    @property
    def n_dimensions(self) -> int:
        return self._lc.dimension

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._lc.n_rows, self._lc.dimension)

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        return self._lc.is_satisfied(x, tol)

    def violations(self, x: np.ndarray) -> List[Tuple[int, str, float]]:
        residuals = self._lc.A @ x - self._lc.b
        return [
            (i, self._lc.row_names[i], float(residuals[i]))
            for i in range(self._lc.n_rows)
            if residuals[i] > 1e-9
        ]

    def slack(self, x: np.ndarray) -> np.ndarray:
        return self._lc.row_slack(x)

    def active_set(self, x: np.ndarray, tol: float = 1e-6) -> List[int]:
        slack = self._lc.b - self._lc.A @ x
        return [i for i, s in enumerate(slack) if abs(s) < tol]

    def as_linear_constraints(self) -> LinearConstraints:
        return self._lc

    def as_region(self) -> FeasibleRegion:
        return FeasibleRegion([self._lc], self.n_dimensions)

    def with_version(self, version: str) -> Polytope:
        p = Polytope.__new__(Polytope)
        p._lc = self._lc
        p._version = version
        return p

    def with_bound(self, index: int, new_bound: float) -> Polytope:
        new_b = self._lc.b.copy()
        if not (0 <= index < self._lc.n_rows):
            raise ValidationError(f"Index {index} out of range [0, {self._lc.n_rows})")
        new_b[index] = new_bound
        return Polytope(
            self._lc.A.copy(), new_b, self.names, self.tags, self._version,
        )

    def with_bounds(self, updates: Dict[int, float]) -> Polytope:
        new_b = self._lc.b.copy()
        for idx, val in updates.items():
            if not (0 <= idx < self._lc.n_rows):
                raise ValidationError(f"Index {idx} out of range [0, {self._lc.n_rows})")
            new_b[idx] = val
        return Polytope(
            self._lc.A.copy(), new_b, self.names, self.tags, self._version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self._lc.A.tolist(),
            "b": self._lc.b.tolist(),
            "names": list(self.names),
            "authorities": list(self.tags),
            "version": self._version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Polytope:
        return cls(
            A=d["A"], b=d["b"],
            names=d.get("names"),
            tags=d.get("tags") or d.get("authorities"),
            version=d.get("version", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> Polytope:
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return f"Polytope(n={self.n_dimensions}, m={self.n_constraints}, version='{self._version}')"


def enforce_action(
    action: Dict[str, Any],
    schema: Schema,
    polytope: Polytope,
    mode: str = "project",
    max_distance: Optional[float] = None,
) -> Tuple[Dict[str, float], EnforcementOutput]:
    """Legacy convenience: enforce constraints on a structured action dict."""
    if polytope.n_dimensions != schema.dimension:
        raise ValidationError(
            f"Polytope has {polytope.n_dimensions} dimensions, schema has {schema.dimension} fields"
        )
    cfg = EnforcementConfig(mode=mode, max_distance=max_distance)
    vec = schema.vectorize(action)
    output = enforce(vec, polytope.as_region(), cfg, schema)
    return schema.devectorize(output.enforced_vector), output


# ═══════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════

ActionSchema = Schema
GCESystem = NumerailSystem
GCEError = NumerailError

