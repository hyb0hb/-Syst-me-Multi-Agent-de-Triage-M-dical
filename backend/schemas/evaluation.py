"""Schémas Pydantic pour le système d'évaluation reproductible MedTriage AI."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DatasetId = Literal[
    "clinical_test",
    "radiology_test",
    "radiology_test_b",
    "simulated_quadrant",
]

SystemId = Literal["system_a", "system_b", "system_c"]


class EvaluationRequest(BaseModel):
    dataset_id: DatasetId = Field(
        default="clinical_test",
        description="Jeu d'évaluation tenu à l'écart (clinical_test, radiology_test, radiology_test_b, simulated_quadrant)",
    )
    systems: List[SystemId] = Field(
        default_factory=lambda: ["system_a", "system_b", "system_c"],
        min_length=1,
        description="Systèmes à comparer sur exactement les mêmes identifiants d'échantillons",
    )
    sample_size: Optional[int] = Field(
        default=None,
        ge=20,
        le=5000,
        description="Taille du sous-échantillon aléatoire reproductible (None = split complet)",
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
        description="Graine aléatoire fixe pour la sélection reproductible et les IC bootstrap",
    )
    n_bootstrap: int = Field(
        default=400,
        ge=100,
        le=2000,
        description="Nombre de tirages bootstrap stratifiés pour les intervalles de confiance à 95 %",
    )


class ExportEvaluationRequest(EvaluationRequest):
    format: Literal["json", "markdown"] = Field(
        default="markdown",
        description="Format d'export du rapport d'évaluation reproductible",
    )
