"""Banc de tests automatisés pour le Système Multi-Agent de Triage Médical.
Utilise le framework standard 'unittest' pour une exécution sans dépendance supplémentaire.

Exécution :
    py -m unittest tests/test_system.py -v
"""

import json
import os
import unittest
from pathlib import Path
import numpy as np
import joblib
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Chemins
import sys
ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import des modules testés
import server
import demo
from backend.services.triage_service import align_clinical_risk, get_clinical_status, process_decision_agent


class TestAgentDecision(unittest.TestCase):
    """Vérification du comportement de l'Agent Décisionnel de Consensus (Régression Logistique + Sécurité SAU)."""

    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "agent_decision" / "model_lr.joblib"
        assert model_path.exists(), f"Le fichier modèle {model_path} est manquant."
        cls.model = joblib.load(model_path)
        if not hasattr(cls.model, "multi_class"):
            cls.model.multi_class = "auto"

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
        """Vérifie l'ordonnancement clinique rigoureux des 4 quadrants pivots et l'absence de sous-triage."""
        # Q00 : Patient sain (risques faibles)
        p_sain = float(self.model.predict_proba([[0.05, 0.05]])[0, 1])
        # Q10 : Risque pulmonaire isolé
        p_pneu = float(self.model.predict_proba([[0.70, 0.05]])[0, 1])
        # Q01 : Risque AVC isolé
        p_avc = float(self.model.predict_proba([[0.05, 0.70]])[0, 1])
        # Q11 : Risque mixte (choc & comorbidités)
        p_mixte = float(self.model.predict_proba([[0.70, 0.70]])[0, 1])

        # 1. Le cas sain doit être non urgent
        self.assertLess(p_sain, 0.15, f"Un patient sain ne doit pas être sur-trié (p={p_sain:.3f}).")

        # 2. Vérification du protocole de décision SAU (Principe de précaution)
        # Un patient avec pneumonie aiguë avérée (p_img = 0.70) DOIT être classé URGENT
        p_final_pneu, p_comp_pneu, niv_pneu, _, _, _, _, _, _ = process_decision_agent(self.model, 0.70, 0.05)
        self.assertEqual(niv_pneu, "URGENT", "La détresse respiratoire aiguë ne doit pas être sous-triée !")
        self.assertGreaterEqual(p_final_pneu, 0.70)

        # Un patient avec suspicion forte d'AVC (p_clin = 0.70) DOIT être classé URGENT
        p_final_avc, p_comp_avc, niv_avc, _, _, _, _, _, _ = process_decision_agent(self.model, 0.05, 0.70)
        self.assertEqual(niv_avc, "URGENT", "L'urgence neuro-vasculaire ne doit pas être sous-triée !")
        self.assertGreaterEqual(p_final_avc, 0.70)

        # 3. Le cas mixte doit être le plus grave de tous
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


class TestFastAPIRestAPI(unittest.TestCase):
    """Vérification fonctionnelle des endpoints REST FastAPI avec TestClient."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_metrics_endpoint(self):
        """Vérifie que GET /api/metrics répond 200 avec les 3 agents et le summary dynamique."""
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("image", data)
        self.assertIn("clinique", data)
        self.assertIn("decision", data)
        # Vérification du summary dynamique pour recall, accuracy, precision, f1, roc_auc
        self.assertIn("summary", data)
        for metric in ["roc_auc", "accuracy", "recall", "precision", "f1"]:
            self.assertIn(metric, data["summary"])
            for agent in ["cnn", "rf", "lr"]:
                self.assertIn(agent, data["summary"][metric])
                self.assertGreaterEqual(data["summary"][metric][agent], 0.0)
        self.assertIn("decision_formula", data)

    def test_presets_endpoint(self):
        """Vérifie que GET /api/presets retourne les 4 cas cliniques."""
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cas_pneumonie", data)
        self.assertIn("cas_avc", data)
        self.assertIn("cas_critique_mixte", data)
        self.assertIn("cas_routine", data)

    def test_history_endpoint(self):
        """Vérifie que GET /api/history retourne une liste conforme."""
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("evaluations", data)
        self.assertIn("total", data)

    def test_triage_validation_error_status_422(self):
        """Vérifie qu'un âge négatif via l'API retourne HTTP 422 Unprocessable Entity."""
        payload = {
            "age": -10.0,
            "glucose": 100.0,
            "bmi": 24.0,
            "sample_image_name": "normal.jpg",
        }
        res = self.client.post("/api/triage", data=payload)
        self.assertEqual(res.status_code, 422)

    def test_agent2_voie1_metrics_and_three_tiers(self):
        """Vérifie le modèle Voie 1, son seuil OOF gelé et la cohérence des 3 paliers."""
        with open(ROOT / "agent_clinique" / "metrics.json", encoding="utf-8") as f:
            met = json.load(f)
        self.assertGreaterEqual(met["roc_auc"], 0.85)
        self.assertIn("threshold", met)
        thresh = float(met["threshold"])
        self.assertAlmostEqual(thresh, 0.65, places=2)
        self.assertEqual(get_clinical_status(0.15, thresh), "Risque clinique faible")
        self.assertEqual(get_clinical_status(0.464, thresh), "Risque clinique modéré (à surveiller)")
        self.assertEqual(get_clinical_status(0.649, thresh), "Risque clinique modéré (à surveiller)")
        self.assertEqual(get_clinical_status(0.65, thresh), "Risque neuro-vasculaire élevé")

    def test_agent2_stateless_row_invariance_and_missing_bmi(self):
        """Vérifie que l'imputation et MissingIndicator gèrent NaN sans fuite ni variance de lot."""
        import pandas as pd
        rf = joblib.load(ROOT / "agent_clinique" / "model_rf.joblib")
        row_ok = pd.DataFrame([{
            "age": 65.0, "avg_glucose_level": 140.0, "bmi": 28.0,
            "hypertension": 1, "heart_disease": 0, "gender": "Male",
            "ever_married": "Yes", "work_type": "Private",
            "Residence_type": "Urban", "smoking_status": "formerly smoked",
        }])
        row_nan = row_ok.copy()
        row_nan["bmi"] = float("nan")
        batch = pd.concat([row_ok, row_nan], ignore_index=True)
        p_batch = rf.predict_proba(batch)[:, 1]
        p_single_ok = rf.predict_proba(row_ok)[0, 1]
        p_single_nan = rf.predict_proba(row_nan)[0, 1]
        self.assertAlmostEqual(float(p_batch[0]), float(p_single_ok), places=6)
        self.assertAlmostEqual(float(p_batch[1]), float(p_single_nan), places=6)
        # Missing BMI raises predicted stroke risk due to learned bmi_missing indicator
        self.assertNotAlmostEqual(float(p_single_ok), float(p_single_nan), places=3)

    def test_all_presets_end_to_end_voie1(self):
        """Vérifie les 4 presets complets avec les 3 agents et le seuil Voie 1 synchronisé."""
        expected = {
            "cas_pneumonie": ("URGENT", "Risque clinique modéré (à surveiller)"),
            "cas_avc": ("URGENT", "Risque neuro-vasculaire élevé"),
            "cas_critique_mixte": ("URGENT", "Risque neuro-vasculaire élevé"),
            "cas_routine": ("NON URGENT", "Risque clinique faible"),
        }
        for key, (exp_triage, exp_clin_statut) in expected.items():
            p = server.PRESETS[key]
            payload = {
                "age": p["age"], "gender": p["gender"], "hypertension": p["hypertension"],
                "heart_disease": p["heart_disease"], "ever_married": p["ever_married"],
                "work_type": p["work_type"], "residence": p["residence"], "glucose": p["glucose"],
                "bmi": p["bmi"], "smoking": p["smoking"],
                "sample_image_name": Path(p["sample_image"]).name,
            }
            res = self.client.post("/api/triage", data=payload)
            self.assertEqual(res.status_code, 200, msg=f"Echec preset {key}: {res.text}")
            data = res.json()
            self.assertEqual(data["triage"]["niveau"], exp_triage, msg=key)
            self.assertEqual(data["agents"]["clinique"]["statut"], exp_clin_statut, msg=key)
            self.assertAlmostEqual(data["agents"]["clinique"]["seuil_utilise"], 0.65, places=2)


class TestAgent3ThresholdAlignmentAndExperiment(unittest.TestCase):
    """Tests de non-régression pour l'alignement de seuil SAU et les artefacts OOF expérimentaux."""

    def test_align_clinical_risk_monotonicity_and_bounds(self):
        """Sous 0.65 rien ne change ; à 0.65 exact le score bascule à 0.70 (URGENT)."""
        self.assertAlmostEqual(align_clinical_risk(0.05, 0.65), 0.05, places=6)
        self.assertAlmostEqual(align_clinical_risk(0.39, 0.65), 0.39, places=6)
        self.assertAlmostEqual(align_clinical_risk(0.649, 0.65), 0.649, places=6)
        self.assertAlmostEqual(align_clinical_risk(0.65, 0.65), 0.70, places=6)
        self.assertGreater(align_clinical_risk(0.66, 0.65), 0.70)
        self.assertLess(align_clinical_risk(0.66, 0.65), align_clinical_risk(0.75, 0.65))
        self.assertAlmostEqual(align_clinical_risk(1.0, 0.65), 1.0, places=6)

    def test_boundary_clinical_065_to_069_no_longer_undertriaged(self):
        """Un AVC détecté par Agent 2 (>= 0.65) avec radio saine ne doit plus être rétrogradé À SURVEILLER."""
        lr = joblib.load(ROOT / "agent_decision" / "model_lr.joblib")
        if not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"
        p_final, _, niv, _, _, _, _, _, _ = process_decision_agent(lr, 0.0033, 0.66, clin_threshold=0.65)
        self.assertGreaterEqual(p_final, 0.70)
        self.assertEqual(niv, "URGENT")

    def test_cli_and_backend_decision_parity(self):
        """Vérifie que demo.py et triage_service.py prennent des décisions strictement cohérentes."""
        exp_lr_path = ROOT / "models" / "experiments" / "agent3_retraining" / "model_lr_oof.joblib"
        lr_path = exp_lr_path if exp_lr_path.exists() else (ROOT / "agent_decision" / "model_lr.joblib")
        lr = joblib.load(lr_path)
        if not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"
        for p_img, p_clin in [(0.0033, 0.0059), (0.0033, 0.66), (0.98, 0.46), (0.98, 0.80)]:
            p_srv, _, niv_srv, _, _, _, _, _, _ = process_decision_agent(lr, p_img, p_clin, clin_threshold=0.65)
            p_cli, niv_cli, _, _, _ = demo.agent_decision(p_img, p_clin, clin_threshold=0.65)
            self.assertAlmostEqual(p_srv, p_cli, places=6)
            self.assertTrue(niv_cli.startswith(niv_srv), msg=f"{niv_cli} vs {niv_srv}")

    def test_experimental_oof_artifacts_integrity(self):
        """Vérifie le modèle expérimental isolé OOF, l'absence de fuite et de faux appariements."""
        import pandas as pd
        exp_dir = ROOT / "models" / "experiments" / "agent3_retraining"
        cfg_path = exp_dir / "training_config.json"
        oof_path = exp_dir / "oof_predictions_agent2_voie1.csv"
        lr_exp_path = exp_dir / "model_lr_oof.joblib"
        self.assertTrue(cfg_path.exists(), f"Configuration expérimentale manquante : {cfg_path}")
        self.assertTrue(oof_path.exists(), f"Prédictions OOF manquantes : {oof_path}")
        self.assertTrue(lr_exp_path.exists(), f"Modèle expérimental manquant : {lr_exp_path}")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        audit = cfg["dataset_audit"]
        self.assertFalse(audit["cross_patient_paired"])
        self.assertEqual(audit["clinical_train_rows"], 4088)
        self.assertEqual(audit["clinical_train_positives"], 199)
        self.assertEqual(audit["clinical_test_rows"], 1022)
        self.assertEqual(audit["clinical_test_positives"], 50)

        oof_df = pd.read_csv(oof_path)
        self.assertEqual(len(oof_df), 4088)
        self.assertEqual(int(oof_df["label"].sum()), 199)

        lr_exp = joblib.load(lr_exp_path)
        c_img, c_clin = float(lr_exp.coef_[0][0]), float(lr_exp.coef_[0][1])
        self.assertGreater(c_img, 1.0)
        self.assertGreater(c_clin, 1.0)
        self.assertLess(max(c_img, c_clin) / min(c_img, c_clin), 2.5)
        self.assertLess(float(lr_exp.predict_proba([[0.05, 0.05]])[0, 1]), 0.15)
        self.assertGreater(float(lr_exp.predict_proba([[0.70, 0.70]])[0, 1]), 0.75)

    def test_experimental_lr_env_override_leaves_prod_intact(self):
        """Vérifie que main charge OOF par défaut et que MEDTRIAGE_LR_PATH permet de basculer sans altérer prod."""
        from backend.services import model_service
        legacy_lr = ROOT / "agent_decision" / "model_lr.joblib"
        prev = os.environ.get("MEDTRIAGE_LR_PATH")
        try:
            os.environ.pop("MEDTRIAGE_LR_PATH", None)
            model_service.MODELS["lr"] = None
            model_service.MODELS_MTIMES.pop("lr", None)
            _, _, default_lr = model_service.get_models()
            self.assertAlmostEqual(float(default_lr.coef_[0][0]), 2.7488, places=3)
            default_met = model_service.load_metrics(force_reload=True)
            self.assertAlmostEqual(float(default_met["coefficients"]["image"]), 2.7488, places=3)
            self.assertIn("2.75", default_met["decision_formula"])

            os.environ["MEDTRIAGE_LR_PATH"] = str(legacy_lr)
            _, _, loaded_lr = model_service.get_models()
            self.assertAlmostEqual(float(loaded_lr.coef_[0][0]), 3.1397, places=3)
        finally:
            if prev is None:
                os.environ.pop("MEDTRIAGE_LR_PATH", None)
            else:
                os.environ["MEDTRIAGE_LR_PATH"] = prev
            model_service.MODELS["lr"] = None
            model_service.MODELS_MTIMES.pop("lr", None)
            model_service.load_metrics(force_reload=True)


if __name__ == "__main__":
    unittest.main()
