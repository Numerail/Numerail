"""Numerail v5.0.0 — Deterministic Geometric Enforcement for AI Actuation Safety.

Usage:
    import numerail as nm

    region = nm.box_constraints([0, 0], [1, 1])
    output = nm.enforce(np.array([1.5, 0.5]), region)
    assert output.result in (nm.EnforcementResult.APPROVE, nm.EnforcementResult.PROJECT)
    assert region.is_feasible(output.enforced_vector)

Layers:
    numerail.engine    — mathematical kernel (enforce, constraints, schema, system)
    numerail.parser    — strict policy parser + lint_config
    numerail.service   — production runtime service
    numerail.local     — in-memory local mode (exercises production code path)
    numerail.protocols — typed Protocol interfaces for production repositories
"""

__version__ = "5.0.0"

# ── Engine core ──────────────────────────────────────────────────────────
from numerail.engine import (
    # Functions
    enforce,
    project,
    box_constraints,
    halfplane,
    combine_regions,
    ellipsoid,
    check_feasibility,
    chebyshev_center,
    synthesize_feedback,
    enforce_action,
    merge_trusted_context,
    # System
    NumerailSystem,
    Schema,
    BudgetSpec,
    AuditChain,
    MetricsCollector,
    RegionVersionStore,
    FeasibleRegion,
    BudgetTracker,
    # Constraints
    ConvexConstraint,
    LinearConstraints,
    QuadraticConstraint,
    SOCPConstraint,
    PSDConstraint,
    # Config and output
    EnforcementResult,
    EnforcementConfig,
    EnforcementOutput,
    RollbackResult,
    DimensionPolicy,
    RoutingDecision,
    RoutingThresholds,
    ProjectionResult,
    NormalizerRange,
    # Exceptions
    NumerailError,
    ValidationError,
    ConstraintError,
    InfeasibleRegionError,
    SolverError,
    SchemaError,
    ResolutionError,
    # Backward compat
    Polytope,
    ActionSchema,
    GCESystem,
    GCEError,
)

# ── Production layer ─────────────────────────────────────────────────────
from numerail.errors import AuthorizationError
from numerail.protocols import (
    TransactionManager,
    AuthorizationService,
    PolicyRepository,
    RuntimeRepository,
    LedgerRepository,
    AuditRepository,
    MetricsRepository,
    OutboxRepository,
    LockedRuntimeHead,
    ServiceRequest,
)
from numerail.parser import PolicyParser, lint_config
from numerail.service import NumerailRuntimeService
from numerail.local import NumerailSystemLocal
