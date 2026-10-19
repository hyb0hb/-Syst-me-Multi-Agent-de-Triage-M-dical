"""Service orchestrateur de triage médical multi-agent."""

import io
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile, status
import numpy as np
import pandas as pd
from PIL import Image

from backend.config import SAMPLES_DIR, CHEST_XRAY_DIR
from backend.services.model_service import get_models, load_metrics, METRICS_CACHE
from backend.services.history_service import save_triage_result


# ---------------------------------------------------------------------------
# Paliers de décision de triage (MTS / ESI — constante, calculée une fois)
# ---------------------------------------------------------------------------
_TRIAGE_LEVELS = [
    (
        0.70, "URGENT", "danger",
        "Prise en charge prioritaire (< 15 min)",
        (
            "Orientation immédiate SAU (Service d'Accueil des Urgences). "
            "Monitorage continu cardio-respiratoire, pose de voie veineuse, bilan biologique d'urgence "
            "et avis spécialisé urgent (neurologie / pneumologie / réanimation selon tableau clinique dominant)."
        ),
    ),
    (
        0.40, "À SURVEILLER", "warning",
        "Consultation prioritaire (< 2 heures)",
        (
            "Admission en unité de surveillance rapprochée. Réalisation d'examens complémentaires "
            "(ECG, gaz du sang, bilan métabolique), surveillance de l'hémodynamique et réévaluation médicale programmée."
        ),
    ),
    (
        0.0, "NON URGENT", "success",
        "Prise en charge ambulatoire standard",
        (
            "Absence de signe de détresse vitale immédiate détecté par les deux filières. "
            "Orientation en consultation de médecine générale ou filière de soins non programmés "
            "avec remise des consignes d'alerte en cas d'apparition de nouveaux symptômes."
        ),
    ),
]

_DISCLAIMER = (
    "DISCLAIMER MÉDICAL RÉGLEMENTAIRE : Ce système est un prototype de recherche pédagogique "
    "et universitaire. Il ne constitue pas un dispositif médical au sens de la réglementation "
    "(marquage CE-IVD / FDA). Les prédictions et scores de triage sont fournis à titre indicatif "
    "et ne doivent en aucun cas se substituer à une décision médicale ou un examen clinique réalisé par un médecin."
)


# ---------------------------------------------------------------------------
# Agent 1 : Vision CNN
# ---------------------------------------------------------------------------
async def process_image_agent(
    cnn_model,
    image: Optional[UploadFile] = None,
    sample_image_name: Optional[str] = None,
) -> float:
    """Agent 1 (Vision CNN MobileNetV2) : inférence sur cliché radiologique."""
    if image and image.filename:
        content = await image.read()
        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image téléversé est vide.")
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Le fichier image dépasse la taille limite autorisée (15 Mo).",
            )
        try:
            buf = io.BytesIO(content)
            Image.open(buf).verify()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fichier image invalide ou corrompu. Format JPEG ou PNG requis.",
            )
        if pil_img.width < 32 or pil_img.height < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dimensions trop faibles ({pil_img.width}x{pil_img.height}). Minimum 32x32 px.",
            )
    elif sample_image_name:
        if sample_image_name.startswith("dataset:"):
            rel = sample_image_name.replace("dataset:", "").replace("\\", "/").strip("/")
            sample_path = (CHEST_XRAY_DIR / rel).resolve()
            if not sample_path.exists() or not str(sample_path).startswith(str(CHEST_XRAY_DIR.resolve())):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cliché radiologique du dataset '{rel}' introuvable.",
                )
        else:
            clean_name = Path(sample_image_name).name
            sample_path = SAMPLES_DIR / clean_name
            if not sample_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image d'exemple '{clean_name}' introuvable.",
                )
        pil_img = Image.open(sample_path).convert("RGB")
    else:
        pil_img = Image.open(SAMPLES_DIR / "normal.jpg").convert("RGB")

    arr = np.expand_dims(
        np.array(pil_img.resize((224, 224)), dtype=np.float32), axis=0
    )
    return float(cnn_model(arr, training=False).numpy()[0][0])


# ---------------------------------------------------------------------------
# Agent 2 : Random Forest
# ---------------------------------------------------------------------------
def process_clinical_agent(
    rf_model,
    age: float, glucose: float, bmi: Optional[float],
    hypertension: int, heart_disease: int,
    gender: str, ever_married: str, work_type: str,
    residence: str, smoking: str,
) -> float:
    """Agent 2 (Random Forest) : inférence sur les données cliniques tabulaires."""
    df_row = pd.DataFrame([{
        "age": age,
        "avg_glucose_level": glucose,
        "bmi": bmi if (bmi is not None and not np.isnan(bmi)) else float("nan"),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "smoking_status": smoking,
    }])
    return float(rf_model.predict_proba(df_row)[0, 1])


# ---------------------------------------------------------------------------
# Agent 3 : Régression Logistique
# ---------------------------------------------------------------------------
def process_decision_agent(lr_model, p_img: float, p_clin: float) -> Tuple:
    """Agent 3 (Consensus Régression Logistique) : synthèse composite de triage."""
    p_final = float(lr_model.predict_proba(np.array([[p_img, p_clin]]))[0, 1])

    # Résolution du palier depuis la constante
    for threshold, niveau, couleur, delai, conduite in _TRIAGE_LEVELS:
        if p_final >= threshold:
            break

    coef_img = float(lr_model.coef_[0][0])
    coef_clin = float(lr_model.coef_[0][1])
    intercept = float(lr_model.intercept_[0])

    return p_final, niveau, couleur, delai, conduite, coef_img, coef_clin, intercept


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------
async def run_full_triage(
    image: Optional[UploadFile],
    sample_image_name: Optional[str],
    age: float, gender: str,
    hypertension: int, heart_disease: int,
    ever_married: str, work_type: str,
    residence: str, glucose: float,
    bmi: Optional[float], smoking: str,
) -> dict:
    """Orchestration de bout en bout des trois agents avec génération du rapport complet."""
    cnn_model, rf_model, lr_model = get_models()

    # Charge les métriques uniquement si le cache est vide (optimisation)
    if not METRICS_CACHE:
        load_metrics()

    # Agent 1 : Vision
    try:
        p_img = await process_image_agent(cnn_model, image, sample_image_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erreur Agent Image (CNN) : {e}")

    # Agent 2 : Clinique
    try:
        p_clin = process_clinical_agent(
            rf_model, age, glucose, bmi, hypertension, heart_disease,
            gender, ever_married, work_type, residence, smoking,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erreur Agent Clinique (Random Forest) : {e}")

    # Agent 3 : Consensus
    try:
        p_final, niveau, couleur, delai, conduite, coef_img, coef_clin, intercept = (
            process_decision_agent(lr_model, p_img, p_clin)
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erreur Agent Décision (Régression Logistique) : {e}")

    img_auc  = METRICS_CACHE.get("image",    {}).get("roc_auc", 0.965)
    clin_auc = METRICS_CACHE.get("clinique", {}).get("roc_auc", 0.846)
    dec_auc  = METRICS_CACHE.get("decision", {}).get("roc_auc", 0.822)

    # Synthèse comorbidités
    comorbidites = [
        *( ["Hypertension artérielle"] if hypertension else []),
        *( ["Cardiopathie"]             if heart_disease else []),
        *( ["Tabagisme actif / passé"]  if smoking in ("smokes", "formerly smoked") else []),
    ] or ["Aucune comorbidité majeure déclarée"]

    result = {
        "triage": {
            "niveau": niveau,
            "score_urgence": round(p_final * 100, 1),
            "couleur": couleur,
            "delai_recommande": delai,
            "conduite_a_tenir": conduite,
        },
        "agents": {
            "image": {
                "nom": "Agent Vision Médicale (CNN MobileNetV2)",
                "risque_pneumonie": round(p_img * 100, 1),
                "statut": "Opacités alvéolaires suspectes" if p_img >= 0.5 else "Champs pulmonaires dégagés",
                "modele": "Transfer Learning MobileNetV2 fine-tuné",
                "auc": round(img_auc, 3),
            },
            "clinique": {
                "nom": "Agent Clinique (Random Forest)",
                "risque_avc": round(p_clin * 100, 1),
                "statut": "Risque neuro-vasculaire élevé" if p_clin >= 0.5 else "Risque clinique modéré / faible",
                "modele": "Random Forest (700 arbres régularisés)",
                "auc": round(clin_auc, 3),
            },
            "decision": {
                "nom": "Agent Décisionnel de Consensus (Régression Logistique)",
                "formule": f"z = ({coef_img:.2f} × P_img) + ({coef_clin:.2f} × P_clin) + ({intercept:.2f})",
                "poids_image": round(coef_img, 2),
                "poids_clinique": round(coef_clin, 2),
                "biais": round(intercept, 2),
                "auc": round(dec_auc, 3),
            },
        },
        "patient_synthese": {
            "age": age,
            "gender": {"Male": "Homme", "Female": "Femme"}.get(gender, "Autre"),
            "glycemie": f"{glucose} mg/dL",
            "imc": f"{bmi:.1f} kg/m²" if bmi else "Non renseigné",
            "comorbidites": comorbidites,
        },
        "disclaimer_medical": _DISCLAIMER,
    }

    # ✅ Sauvegarde dans l'historique JSON (CORRECTEMENT placée avant le return)
    patient_info = {
        "age": age, "gender": gender, "glucose": glucose, "bmi": bmi,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type,
        "residence": residence, "smoking": smoking,
    }
    result["eval_id"] = save_triage_result(result, patient_info)

    return result
