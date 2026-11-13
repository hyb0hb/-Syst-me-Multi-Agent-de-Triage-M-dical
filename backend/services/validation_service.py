"""Service de validation clinique des paramètres physiologiques d'entrée."""

from typing import Optional
from fastapi import HTTPException, status


# Compatibilité Starlette/FastAPI toutes versions
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)


def _reject(detail: str) -> None:
    """Lève une erreur HTTP 422 avec le message donné."""
    raise HTTPException(status_code=HTTP_422, detail=detail)


def _check_enum(value: str, allowed: tuple, label: str) -> None:
    if value not in allowed:
        _reject(f"{label} '{value}' invalide. Valeurs autorisées : {', '.join(allowed)}.")


def _check_range(value: float, lo: float, hi: float, label: str, unit: str = "") -> None:
    if value < lo or value > hi:
        _reject(f"{label} aberrant ({value}{unit}). Plage admissible : {lo} – {hi}{unit}.")


def validate_clinical_inputs(
    age: float,
    glucose: float,
    bmi: Optional[float],
    hypertension: int,
    heart_disease: int,
    gender: str,
    ever_married: str,
    work_type: str,
    residence: str,
    smoking: str,
) -> None:
    """Valide les plages physiologiques et catégories pour prévenir les entrées aberrantes."""
    _check_range(age, 0, 125, "Âge", " ans")
    _check_range(glucose, 20, 800, "Glycémie moyenne", " mg/dL")
    if bmi is not None:
        _check_range(bmi, 8, 100, "Indice de masse corporelle (IMC)", " kg/m²")

    if hypertension not in (0, 1):
        _reject("Le statut d'hypertension doit être 0 (Non) ou 1 (Oui).")
    if heart_disease not in (0, 1):
        _reject("Le statut de cardiopathie doit être 0 (Non) ou 1 (Oui).")

    _check_enum(gender, ("Male", "Female", "Other"), "Genre")
    _check_enum(ever_married, ("Yes", "No"), "Statut marital")
    _check_enum(work_type, ("Private", "Self-employed", "Govt_job", "children", "Never_worked"), "Type d'activité")
    _check_enum(residence, ("Urban", "Rural"), "Lieu de résidence")
    _check_enum(smoking, ("formerly smoked", "never smoked", "smokes", "Unknown"), "Statut tabagique")
