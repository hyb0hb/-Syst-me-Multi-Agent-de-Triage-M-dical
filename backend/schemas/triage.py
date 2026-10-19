"""Schémas Pydantic pour la validation et la sérialisation des données de triage."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TriageEvaluationResponse(BaseModel):
    """Structure de réponse complète retournée par l'orchestrateur de triage."""

    niveau: str = Field(..., description="Niveau de triage : URGENT, À SURVEILLER, NON URGENT")
    score_urgence: float = Field(..., description="Probabilité d'urgence composite (0 à 100 %)")
    couleur: str = Field(..., description="Badge CSS : danger, warning, success")
    delai_recommande: str = Field(..., description="Délai clinique recommandé de prise en charge")
    conduite_a_tenir: str = Field(..., description="Protocole d'orientation médicale et examens")


class AgentImageResult(BaseModel):
    nom: str
    risque_pneumonie: float
    statut: str
    modele: str
    auc: float


class AgentCliniqueResult(BaseModel):
    nom: str
    risque_avc: float
    statut: str
    modele: str
    auc: float


class AgentDecisionResult(BaseModel):
    nom: str
    formule: str
    poids_image: float
    poids_clinique: float
    biais: float
    auc: float


class AgentsSummary(BaseModel):
    image: AgentImageResult
    clinique: AgentCliniqueResult
    decision: AgentDecisionResult


class PatientSynthese(BaseModel):
    age: float
    gender: str
    glycemie: str
    imc: str
    comorbidites: List[str]


class TriageResponse(BaseModel):
    triage: TriageEvaluationResponse
    agents: AgentsSummary
    patient_synthese: PatientSynthese
    disclaimer_medical: str
