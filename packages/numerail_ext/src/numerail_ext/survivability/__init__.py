"""Numerail survivability extension — supervisory degradation around the V5 kernel."""

from .types import (
    BreakerMode,
    BreakerThresholds,
    BreakerDecision,
    TelemetrySnapshot,
    WorkloadRequest,
    TransitionEnvelope,
    ExecutableGrant,
    ExecutionReceipt,
    GovernedStep,
    TransitionModel,
    ReservationManager,
    Digestor,
    NumerailBackend,
)
from .breaker import BreakerStateMachine
from .transition_model import IncidentCommanderTransitionModel
from .policy_builder import build_v5_policy_from_envelope
from .validation import ReceiptValidationError, validate_receipt_against_grant
from .governor import StateTransitionGovernor
from .local_backend import LocalNumerailBackend
from .global_default import build_global_default
from .contract import NumerailPolicyContract
from .hitl import (
    HumanReviewProfile,
    HumanReviewTriggers,
    ReviewMode,
    PendingAction,
    SupervisedStepResult,
    SupervisedGovernor,
)
from .local_gateway import LocalApprovalGateway
