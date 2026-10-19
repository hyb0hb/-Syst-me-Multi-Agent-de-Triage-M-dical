"""Package des schémas Pydantic du backend."""

from backend.schemas.triage import (
    AgentCliniqueResult,
    AgentDecisionResult,
    AgentImageResult,
    AgentsSummary,
    PatientSynthese,
    TriageEvaluationResponse,
    TriageResponse,
)

__all__ = [
    "AgentCliniqueResult",
    "AgentDecisionResult",
    "AgentImageResult",
    "AgentsSummary",
    "PatientSynthese",
    "TriageEvaluationResponse",
    "TriageResponse",
]
