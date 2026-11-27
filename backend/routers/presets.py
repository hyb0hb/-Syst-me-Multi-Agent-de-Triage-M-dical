"""Router FastAPI fournissant les profils cliniques préconfigurés et les cas aléatoires du dataset."""

import random
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from backend.config import PRESETS, STROKE_CSV, CHEST_XRAY_DIR

router = APIRouter(prefix="/api", tags=["Presets & Dataset Cas Aléatoires"])

# Cache en mémoire pour réponses instantanées (< 5ms) — strictement limité au jeu de test !
_stroke_test_df: Optional[pd.DataFrame] = None
_normal_imgs = None
_pneu_imgs = None


def _get_dataset_resources():
    global _stroke_test_df, _normal_imgs, _pneu_imgs
    if _stroke_test_df is None and STROKE_CSV.exists():
        df = pd.read_csv(STROKE_CSV)
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        _, _stroke_test_df = train_test_split(
            df, test_size=0.2, stratify=df["stroke"].astype(int), random_state=42
        )
        _stroke_test_df = _stroke_test_df.reset_index(drop=True)

    if _normal_imgs is None and (CHEST_XRAY_DIR / "test" / "NORMAL").exists():
        _normal_imgs = sorted(
            list((CHEST_XRAY_DIR / "test" / "NORMAL").glob("*.jpeg")) + list((CHEST_XRAY_DIR / "test" / "NORMAL").glob("*.jpg")),
            key=lambda p: p.name,
        )
        _pneu_imgs = sorted(
            list((CHEST_XRAY_DIR / "test" / "PNEUMONIA").glob("*.jpeg")) + list((CHEST_XRAY_DIR / "test" / "PNEUMONIA").glob("*.jpg")),
            key=lambda p: p.name,
        )

    return _stroke_test_df, _normal_imgs, _pneu_imgs


@router.get(
    "/presets",
    summary="Récupération des cas cliniques préconfigurés",
    description="Retourne la liste des profils types (Pneumonie, AVC, Cas Mixte, Ambulatoire) pour tests et démo.",
)
def get_presets():
    return PRESETS


@router.get(
    "/random-case",
    summary="Tirage d'un patient et d'une radiographie aléatoires du jeu de TEST uniquement",
    description="Sélectionne un profil patient dans le split test clinique (N=1022) et une radio dans chest_xray/test (graine fixe optionnelle).",
)
def get_random_case(
    quadrant: str = Query("any", description="Cibler un quadrant: any, Q00, Q10, Q01, Q11"),
    seed: Optional[int] = Query(None, ge=0, le=2_147_483_647, description="Graine aléatoire fixe optionnelle pour reproductibilité"),
):
    df, normals, pneus = _get_dataset_resources()
    if df is None or not normals or not pneus:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Les fichiers du dataset local (data/) ne sont pas disponibles sur le serveur."
        )

    rng = random.Random(seed)

    # Filtrage selon le quadrant pivot désiré (strictement dans held-out test!)
    if quadrant in ("Q01", "Q11"):
        sub_df = df[df["stroke"] == 1]
    elif quadrant in ("Q00", "Q10"):
        sub_df = df[df["stroke"] == 0]
    else:
        sub_df = df

    chosen_idx = rng.randrange(len(sub_df))
    row = sub_df.iloc[chosen_idx]

    # Sélection de la radiographie
    if quadrant in ("Q10", "Q11"):
        img = rng.choice(pneus)
        img_label = "PNEUMONIA"
    elif quadrant in ("Q00", "Q01"):
        img = rng.choice(normals)
        img_label = "NORMAL"
    else:
        if rng.random() < 0.5:
            img = rng.choice(normals)
            img_label = "NORMAL"
        else:
            img = rng.choice(pneus)
            img_label = "PNEUMONIA"

    rel_path = f"test/{img_label}/{img.name}"
    stroke_val = int(row["stroke"])
    # Préserver les valeurs manquantes d'IMC sans imputation silencieuse (cohérent avec MissingIndicator)
    bmi_val = round(float(row["bmi"]), 1) if not pd.isna(row["bmi"]) else None

    return {
        "source": "Kaggle Held-Out Test Splits Only (X_test=1022, chest_xray/test=624)",
        "held_out_split": True,
        "cross_patient_paired": False,
        "seed_used": seed,
        "patient_id": int(row["id"]),
        "age": float(row["age"]),
        "gender": str(row["gender"]) if str(row["gender"]) in ("Male", "Female", "Other") else "Male",
        "glucose": round(float(row["avg_glucose_level"]), 1),
        "bmi": bmi_val,
        "hypertension": int(row["hypertension"]),
        "heart_disease": int(row["heart_disease"]),
        "ever_married": str(row["ever_married"]) if str(row["ever_married"]) in ("Yes", "No") else "Yes",
        "work_type": str(row["work_type"]) if str(row["work_type"]) in ("Private", "Self-employed", "Govt_job", "children", "Never_worked") else "Private",
        "residence": str(row["Residence_type"]) if str(row["Residence_type"]) in ("Urban", "Rural") else "Urban",
        "smoking": str(row["smoking_status"]) if str(row["smoking_status"]) in ("formerly smoked", "never smoked", "smokes", "Unknown") else "never smoked",
        "true_stroke": stroke_val,
        "stroke_label": "AVC Avéré (stroke=1)" if stroke_val == 1 else "Pas d'AVC (stroke=0)",
        "image_filename": img.name,
        "image_class": img_label,
        "sample_token": f"dataset:{rel_path}",
        "image_url": f"/api/dataset-image?path={rel_path}",
        "ground_truth_summary": f"Test Patient #{int(row['id'])} ({'AVC' if stroke_val==1 else 'Non-AVC'}) · Radio {img_label} ({img.name}) [Unpaired]",
    }


@router.get(
    "/dataset-image",
    summary="Récupération du flux image d'une radio du dataset",
    description="Sert le fichier radiologique JPEG directement depuis le dossier data/chest_xray/.",
)
def get_dataset_image(path: str = Query(..., description="Chemin relatif dans chest_xray")):
    rel = path.replace("\\", "/").strip("/")
    full_path = (CHEST_XRAY_DIR / rel).resolve()
    if not full_path.exists() or not str(full_path).startswith(str(CHEST_XRAY_DIR.resolve())):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image du dataset '{rel}' introuvable ou chemin non autorisé."
        )
    return FileResponse(full_path, media_type="image/jpeg")
