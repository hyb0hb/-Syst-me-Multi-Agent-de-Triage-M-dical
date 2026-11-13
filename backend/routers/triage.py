"""Router FastAPI pour l'évaluation et l'orchestration du triage médical."""

from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile

from backend.schemas.triage import TriageResponse
from backend.services.triage_service import run_full_triage
from backend.services.validation_service import validate_clinical_inputs

router = APIRouter(prefix="/api", tags=["Triage Médical Multi-Agent"])


@router.post(
    "/triage",
    response_model=TriageResponse,
    summary="Exécution du protocole de triage multi-agent",
    description=(
        "Coordonne l'analyse simultanée de l'imagerie thoracique (CNN) "
        "et des facteurs de risque neuro-vasculaires (Random Forest) "
        "puis calcule la délibération de consensus (Régression Logistique)."
    ),
)
async def perform_triage(
    image: Optional[UploadFile] = File(None, description="Cliché radiologique (JPEG/PNG)"),
    sample_image_name: Optional[str] = Form(None, description="Nom de l'image de test préchargée"),
    age: float = Form(..., description="Âge du patient (années)", examples=[54.0]),
    gender: str = Form("Male", description="Genre (Male, Female, Other)"),
    hypertension: int = Form(0, description="Antécédent d'hypertension artérielle (0 ou 1)"),
    heart_disease: int = Form(0, description="Antécédent de cardiopathie (0 ou 1)"),
    ever_married: str = Form("Yes", description="Statut marital (Yes ou No)"),
    work_type: str = Form("Private", description="Type d'activité professionnelle"),
    residence: str = Form("Urban", description="Lieu de résidence (Urban ou Rural)"),
    glucose: float = Form(..., description="Glycémie moyenne à jeun (mg/dL)", examples=[98.4]),
    bmi: Optional[float] = Form(None, description="Indice de masse corporelle (kg/m²)", examples=[26.2]),
    smoking: str = Form("never smoked", description="Statut tabagique"),
):
    # Validation clinique stricte des entrées
    validate_clinical_inputs(
        age=age,
        glucose=glucose,
        bmi=bmi,
        hypertension=hypertension,
        heart_disease=heart_disease,
        gender=gender,
        ever_married=ever_married,
        work_type=work_type,
        residence=residence,
        smoking=smoking,
    )

    # Exécution du triage de bout en bout
    return await run_full_triage(
        image=image,
        sample_image_name=sample_image_name,
        age=age,
        gender=gender,
        hypertension=hypertension,
        heart_disease=heart_disease,
        ever_married=ever_married,
        work_type=work_type,
        residence=residence,
        glucose=glucose,
        bmi=bmi,
        smoking=smoking,
    )
