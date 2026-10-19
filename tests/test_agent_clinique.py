"""Tests unitaires et de validation pour l'Agent 2 : Clinique (Random Forest).

Exécution :
    py -m unittest tests/test_agent_clinique.py -v
"""

import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "agent_clinique" / "model_rf.joblib"
METRICS_PATH = ROOT / "agent_clinique" / "metrics.json"


class TestAgentClinique(unittest.TestCase):
    """Banc de tests dédié à l'Agent 2 (Random Forest - Risque AVC)."""

    @classmethod
    def setUpClass(cls):
        assert MODEL_PATH.exists(), f"Modèle introuvable : {MODEL_PATH}"
        assert METRICS_PATH.exists(), f"Fichier métriques introuvable : {METRICS_PATH}"
        cls.model = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            cls.metrics = json.load(f)

    def test_model_loaded(self):
        """Vérifie que le modèle Random Forest est correctement instancié et chargé."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, "predict_proba"))
        self.assertTrue(hasattr(self.model, "classes_"))

    def test_metrics_standards(self):
        """Vérifie que les métriques certifiées respectent le cahier des charges (AUC >= 0.84, Rappel >= 0.75)."""
        auc = self.metrics.get("roc_auc", 0.0)
        self.assertGreaterEqual(auc, 0.84, f"L'AUC ({auc}) doit être supérieure ou égale à 0.84.")
        report = self.metrics.get("report", {})
        recall_stroke = report.get("1", {}).get("recall", 0.0)
        self.assertGreaterEqual(recall_stroke, 0.75, f"Le rappel sur l'AVC ({recall_stroke}) doit être >= 0.75.")

    def test_healthy_patient_prediction(self):
        """Vérifie qu'un profil sain (jeune, sans comorbidités) a une faible probabilité d'AVC."""
        patient_sain = pd.DataFrame([{
            "gender": "Female",
            "age": 25.0,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 85.0,
            "bmi": 21.5,
            "smoking_status": "never smoked"
        }])
        prob = float(self.model.predict_proba(patient_sain)[0, 1])
        self.assertLess(prob, 0.20, f"Un jeune patient sain ne doit pas avoir un risque élevé (obtenu: {prob:.3f}).")

    def test_high_risk_patient_prediction(self):
        """Vérifie qu'un patient à risque élevé (âgé, diabétique, hypertendu) a une forte probabilité d'AVC."""
        patient_risque = pd.DataFrame([{
            "gender": "Male",
            "age": 78.0,
            "hypertension": 1,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Self-employed",
            "Residence_type": "Rural",
            "avg_glucose_level": 220.0,
            "bmi": 32.0,
            "smoking_status": "formerly smoked"
        }])
        prob = float(self.model.predict_proba(patient_risque)[0, 1])
        self.assertGreater(prob, 0.50, f"Un patient à haut risque doit dépasser 0.50 (obtenu: {prob:.3f}).")

    def test_risk_monotonicity_age(self):
        """Vérifie que le risque d'AVC augmente avec l'âge à profil égal."""
        pt_30 = pd.DataFrame([{
            "gender": "Male", "age": 30.0, "hypertension": 0, "heart_disease": 0,
            "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 95.0, "bmi": 25.0, "smoking_status": "never smoked"
        }])
        pt_75 = pd.DataFrame([{
            "gender": "Male", "age": 75.0, "hypertension": 0, "heart_disease": 0,
            "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 95.0, "bmi": 25.0, "smoking_status": "never smoked"
        }])
        p_30 = float(self.model.predict_proba(pt_30)[0, 1])
        p_75 = float(self.model.predict_proba(pt_75)[0, 1])
        self.assertGreater(p_75, p_30, f"Le risque à 75 ans ({p_75:.3f}) doit dépasser celui à 30 ans ({p_30:.3f}).")


if __name__ == "__main__":
    unittest.main()
