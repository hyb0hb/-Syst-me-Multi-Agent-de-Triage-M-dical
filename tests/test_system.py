"""Banc de tests automatisés pour le Système Multi-Agent de Triage Médical.
Utilise le framework standard 'unittest' pour une exécution sans dépendance supplémentaire.

Exécution :
    py -m unittest tests/test_system.py -v
"""

import json
import unittest
from pathlib import Path
import numpy as np
import joblib
from fastapi import HTTPException

# Chemins
import sys
ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import des modules testés
import server
import demo


class TestAgentDecision(unittest.TestCase):
    """Vérification du comportement de l'Agent Décisionnel de Consensus (Régression Logistique)."""

    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "agent_decision" / "model_lr.joblib"
        cls.assertTrue(model_path.exists(), f"Le fichier modèle {model_path} est manquant.")
        cls.model = joblib.load(model_path)

    def test_model_coefficients_balance(self):
        """Vérifie que les coefficients radio et clinique sont tous deux positifs et équilibrés."""
        coef_img = self.model.coef_[0][0]
        coef_clin = self.model.coef_[0][1]

        # Les deux coefficients doivent être strictement positifs
        self.assertGreater(coef_img, 1.0, "Le poids de l'image doit être significatif.")
        self.assertGreater(coef_clin, 1.0, "Le poids clinique doit être significatif.")

        # Le ratio ne doit plus être de 6.6 comme dans la version non corrigée
        ratio = max(coef_img, coef_clin) / min(coef_img, coef_clin)
        self.assertLess(ratio, 2.5, f"Déséquilibre excessif entre modalités (ratio={ratio:.2f}).")

    def test_four_clinical_quadrants(self):
        """Vérifie l'ordonnancement clinique rigoureux des 4 quadrants pivots."""
        # Q00 : Patient sain (risques faibles)
        p_sain = float(self.model.predict_proba([[0.05, 0.05]])[0, 1])
        # Q10 : Risque pulmonaire isolé
        p_pneu = float(self.model.predict_proba([[0.70, 0.05]])[0, 1])
        # Q01 : Risque AVC isolé (ne doit plus être écrasé !)
        p_avc = float(self.model.predict_proba([[0.05, 0.70]])[0, 1])
        # Q11 : Risque mixte (choc & comorbidités)
        p_mixte = float(self.model.predict_proba([[0.70, 0.70]])[0, 1])

        # 1. Le cas sain doit être non urgent
        self.assertLess(p_sain, 0.15, f"Un patient sain ne doit pas être sur-trié (p={p_sain:.3f}).")

        # 2. L'AVC isolé DOIT déclencher une alerte (> 40% seuil de surveillance/urgence)
        self.assertGreaterEqual(p_avc, 0.40, f"L'AVC isolé est écrasé à tort (p={p_avc:.3f}) !")

        # 3. La pneumonie isolée doit déclencher un palier d'alerte
        self.assertGreaterEqual(p_pneu, 0.25, f"La pneumonie isolée est sous-évaluée (p={p_pneu:.3f}).")

        # 4. Le cas mixte doit être le plus grave de tous
        self.assertGreater(p_mixte, p_pneu, "Le cas mixte doit dépasser la pneumonie seule.")
        self.assertGreater(p_mixte, p_avc, "Le cas mixte doit dépasser l'AVC seul.")
        self.assertGreater(p_mixte, 0.75, f"Le cas mixte doit être en urgence vitale (p={p_mixte:.3f}).")

    def test_monotonicity(self):
        """Vérifie que la probabilité d'urgence est strictement monotone croissante."""
        p1 = float(self.model.predict_proba([[0.2, 0.2]])[0, 1])
        p2 = float(self.model.predict_proba([[0.5, 0.2]])[0, 1])
        p3 = float(self.model.predict_proba([[0.5, 0.6]])[0, 1])
        p4 = float(self.model.predict_proba([[0.8, 0.8]])[0, 1])

        self.assertLess(p1, p2)
        self.assertLess(p2, p3)
        self.assertLess(p3, p4)


class TestClinicalInputValidation(unittest.TestCase):
    """Vérification du filtrage et de la robustesse des entrées dans server.py."""

    def test_valid_parameters_pass(self):
        """Vérifie que des paramètres physiologiques normaux sont acceptés sans exception."""
        try:
            server.validate_clinical_inputs(
                age=45.0, glucose=110.0, bmi=24.5, hypertension=0, heart_disease=0,
                gender="Female", ever_married="Yes", work_type="Private",
                residence="Urban", smoking="never smoked"
            )
        except HTTPException:
            self.fail("validate_clinical_inputs a levé une exception inattendue pour des valeurs valides.")

    def test_rejection_negative_age(self):
        """Rejet strict d'un âge négatif."""
        with self.assertRaises(HTTPException) as ctx:
            server.validate_clinical_inputs(
                age=-5.0, glucose=100.0, bmi=22.0, hypertension=0, heart_disease=0,
                gender="Male", ever_married="No", work_type="Private",
                residence="Urban", smoking="never smoked"
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_rejection_aberrant_glucose(self):
        """Rejet strict d'une glycémie impossible (ex: 950 mg/dL ou négative)."""
        with self.assertRaises(HTTPException) as ctx:
            server.validate_clinical_inputs(
                age=50.0, glucose=950.0, bmi=25.0, hypertension=0, heart_disease=0,
                gender="Male", ever_married="Yes", work_type="Private",
                residence="Urban", smoking="never smoked"
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_rejection_aberrant_bmi(self):
        """Rejet strict d'un IMC aberrant (< 10 ou > 100)."""
        with self.assertRaises(HTTPException) as ctx:
            server.validate_clinical_inputs(
                age=30.0, glucose=90.0, bmi=150.0, hypertension=0, heart_disease=0,
                gender="Female", ever_married="No", work_type="Private",
                residence="Urban", smoking="never smoked"
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_rejection_invalid_category(self):
        """Rejet d'une catégorie interdite ou injectée."""
        with self.assertRaises(HTTPException) as ctx:
            server.validate_clinical_inputs(
                age=30.0, glucose=90.0, bmi=22.0, hypertension=0, heart_disease=0,
                gender="Alien", ever_married="No", work_type="Private",
                residence="Urban", smoking="never smoked"
            )
        self.assertEqual(ctx.exception.status_code, 422)


class TestPresetsAndMetricsIntegrity(unittest.TestCase):
    """Contrôle d'intégrité des cas cliniques préenregistrés et des métriques."""

    def test_presets_validity(self):
        """Vérifie que les 4 presets déclarés sont médicalement cohérents et exécutables."""
        presets = server.PRESETS
        self.assertEqual(len(presets), 4, "Il doit y avoir exactement 4 cas cliniques types.")

        for key, p in presets.items():
            self.assertIn("age", p)
            self.assertIn("glucose", p)
            self.assertIn("sample_image", p)
            # Vérifie que l'image associée existe physiquement
            img_file = STATIC_DIR / "samples" / Path(p["sample_image"]).name
            self.assertTrue(img_file.exists(), f"L'image d'exemple pour {key} n'existe pas : {img_file}")

    def test_metrics_json_integrity(self):
        """Vérifie l'existence et la validité des fichiers metrics.json pour chaque agent."""
        agents = ["agent_image", "agent_clinique", "agent_decision"]
        for agent in agents:
            m_path = ROOT / agent / "metrics.json"
            self.assertTrue(m_path.exists(), f"metrics.json manquant pour {agent}")
            with open(m_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("roc_auc", data, f"roc_auc manquant dans {m_path}")
            auc = data["roc_auc"]
            self.assertGreaterEqual(auc, 0.70, f"AUC anormalement basse ({auc}) pour {agent}")
            self.assertLessEqual(auc, 1.0, f"AUC impossible (>1) pour {agent}")


class TestDemoCLIValidation(unittest.TestCase):
    """Vérification des contrôles de validation du script demo.py."""

    def test_demo_image_not_found(self):
        """Vérifie que demo.py lève une ValueError explicite si l'image n'existe pas."""
        from argparse import Namespace
        dummy_args = Namespace(
            age=40.0, glucose=100.0, bmi=25.0, hypertension=0, heart_disease=0,
            gender="Male", ever_married="Yes", work_type="Private",
            residence="Urban", smoking="never smoked"
        )
        fake_path = ROOT / "static" / "samples" / "image_inexistante_999.jpg"
        with self.assertRaises(ValueError) as ctx:
            demo.validate_inputs(dummy_args, fake_path)
        self.assertIn("introuvable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
