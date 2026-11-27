"""Tests automatisés pour le système d'évaluation reproductible MedTriage AI.

Vérifie :
- Séparation stricte train / test sans fuite d'identifiants ni d'étiquettes
- Préservation des valeurs manquantes d'IMC (`NaN` / `None`)
- Exactitude des calculs de matrices de confusion, IC bootstrap et test de McNemar
- Suppression automatique des sous-groupes à effectif insuffisant
- Reproductibilité exacte avec graine fixe et identifiants partagés entre systèmes
- Contrôle d'autorisation sur /api/evaluation/* et export Markdown/JSON
- Préservation des modèles de production
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from agent_clinique.train_rf import FEATURES
from backend.routers.evaluation import get_expected_eval_token
from backend.services.evaluation_service import (
    _exact_mcnemar_p,
    _extract_clinical_features,
    compute_classification_metrics,
    get_clinical_heldout_split,
    get_radiology_heldout_split,
    run_reproducible_evaluation,
)


class TestDataSeparationAndLeakage(unittest.TestCase):
    def test_clinical_and_radiology_split_disjointness(self):
        df = pd.read_csv(ROOT / "data" / "healthcare-dataset-stroke-data.csv")
        train_df, test_df_ref = train_test_split(df, test_size=0.2, stratify=df["stroke"].astype(int), random_state=42)
        eval_df, test_ids, y_test, info = get_clinical_heldout_split()

        self.assertEqual(len(eval_df), 1022)
        self.assertEqual(int(y_test.sum()), 50)
        self.assertFalse(info["cross_patient_paired"])
        self.assertFalse(info["is_simulated"])
        train_ids = set(train_df["id"].astype(int).tolist())
        self.assertEqual(len(train_ids.intersection(set(test_ids.tolist()))), 0)
        self.assertEqual(set(test_ids.tolist()), set(test_df_ref["id"].astype(int).tolist()))

        rad_full, _, _ = get_radiology_heldout_split(split_b_only=False)
        rad_b, _, _ = get_radiology_heldout_split(split_b_only=True)
        self.assertEqual(len(rad_full), 624)
        self.assertEqual(len(rad_b), 312)
        ids_full = [r["sample_id"] for r in rad_full]
        ids_b = set(r["sample_id"] for r in rad_b)
        self.assertTrue(ids_b.issubset(set(ids_full)))

    def test_feature_firewall_strips_labels_and_preserves_nan_bmi(self):
        eval_df, _, _, _ = get_clinical_heldout_split()
        X_feat = _extract_clinical_features(eval_df)
        self.assertEqual(list(X_feat.columns), FEATURES)
        self.assertNotIn("stroke", X_feat.columns)
        self.assertNotIn("id", X_feat.columns)
        self.assertGreater(int(X_feat["bmi"].isna().sum()), 0)

    def test_production_models_unmodified(self):
        lr_hash = hashlib.sha256((ROOT / "agent_decision" / "model_lr.joblib").read_bytes()).hexdigest()[:16]
        rf_hash = hashlib.sha256((ROOT / "agent_clinique" / "model_rf.joblib").read_bytes()).hexdigest()[:16]
        self.assertEqual(lr_hash, "bd10359c545a89c7")
        self.assertEqual(rf_hash, "37b9b0b34114d538")


class TestEvaluationStatistics(unittest.TestCase):
    def test_metrics_confusion_and_ci_calculations(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.85, 0.4, 0.75, 0.2, 0.95])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.70, n_bootstrap=100, seed=42)
        # Negatives (idx 0,1,2,3,8): 4 TN, 1 FP (0.8)
        # Positives (idx 4,5,6,7,9): 4 TP, 1 FN (0.4)
        self.assertEqual(m["confusion_matrix"], [[4, 1], [1, 4]])
        self.assertAlmostEqual(m["sensitivity"], 0.8, places=4)
        self.assertAlmostEqual(m["specificity"], 0.8, places=4)
        self.assertAlmostEqual(m["precision"], 0.8, places=4)
        self.assertAlmostEqual(m["f1"], 0.8, places=4)
        for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "brier_score"]:
            lo, hi = m["ci_95"][k]
            self.assertLessEqual(lo, m[k] + 1e-6)
            self.assertGreaterEqual(hi, m[k] - 1e-6)

    def test_exact_mcnemar_properties(self):
        self.assertAlmostEqual(_exact_mcnemar_p(5, 5), 1.0, places=4)
        self.assertLess(_exact_mcnemar_p(12, 0), 0.001)
        self.assertAlmostEqual(_exact_mcnemar_p(0, 0), 1.0, places=4)

    def test_seeded_reproducibility_subgroups_and_alignment(self):
        r1 = run_reproducible_evaluation("clinical_test", ["system_a", "system_b", "system_c"], sample_size=250, seed=123, n_bootstrap=100)
        r2 = run_reproducible_evaluation("clinical_test", ["system_a", "system_b", "system_c"], sample_size=250, seed=123, n_bootstrap=100)
        self.assertEqual(r1["dataset"]["sample_ids_hash"], r2["dataset"]["sample_ids_hash"])
        self.assertEqual(r1["dataset"]["sample_ids_preview"], r2["dataset"]["sample_ids_preview"])
        self.assertEqual(r1["systems"]["system_c"]["metrics"]["confusion_matrix"], r2["systems"]["system_c"]["metrics"]["confusion_matrix"])
        self.assertEqual(r1["systems"]["system_c"]["metrics"]["roc_auc"], r2["systems"]["system_c"]["metrics"]["roc_auc"])

        # Check full split subgroup suppression (`bmi_missing` only has 3 positives < 5)
        full_c = run_reproducible_evaluation("clinical_test", ["system_c"], sample_size=None, seed=42, n_bootstrap=100)
        sub_map = {s["subgroup_id"]: s for s in full_c["systems"]["system_c"]["subgroups"]}
        self.assertFalse(sub_map["bmi_missing"]["supported"])
        self.assertTrue(sub_map["age_ge_60"]["supported"])


class TestEvaluationAPIAndRandomHeldOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.token = get_expected_eval_token()

    def test_random_case_held_out_and_seeded(self):
        _, heldout_ids, _, _ = get_clinical_heldout_split()
        heldout_set = set(heldout_ids.tolist())
        res1 = self.client.get("/api/random-case?seed=77")
        res2 = self.client.get("/api/random-case?seed=77")
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        d1, d2 = res1.json(), res2.json()
        self.assertTrue(d1["held_out_split"])
        self.assertFalse(d1["cross_patient_paired"])
        self.assertIn(d1["patient_id"], heldout_set)
        self.assertEqual(d1["patient_id"], d2["patient_id"])
        self.assertEqual(d1["image_filename"], d2["image_filename"])

    def test_evaluation_endpoints_require_authorization(self):
        self.assertEqual(self.client.get("/api/evaluation/metadata").status_code, 403)
        self.assertEqual(self.client.get("/api/evaluation/metadata", headers={"X-Eval-Token": "wrong"}).status_code, 403)
        self.assertEqual(self.client.post("/api/evaluation/run", json={"dataset_id": "clinical_test"}).status_code, 403)

    def test_evaluation_run_and_exports_authorized(self):
        hdr = {"X-Eval-Token": self.token}
        meta = self.client.get("/api/evaluation/metadata", headers=hdr)
        self.assertEqual(meta.status_code, 200)
        self.assertEqual(len(meta.json()["datasets"]), 4)

        run_res = self.client.post(
            "/api/evaluation/run",
            headers=hdr,
            json={"dataset_id": "clinical_test", "systems": ["system_a", "system_b", "system_c"], "seed": 42, "n_bootstrap": 120},
        )
        self.assertEqual(run_res.status_code, 200)
        body = run_res.json()
        self.assertEqual(body["dataset"]["evaluated_count"], 1022)
        self.assertIn("system_a", body["systems"])
        self.assertIn("system_c", body["systems"])
        self.assertGreater(body["systems"]["system_c"]["metrics"]["sensitivity"], body["systems"]["system_a"]["metrics"]["sensitivity"])

        exp_md = self.client.post(
            "/api/evaluation/export",
            headers=hdr,
            json={"dataset_id": "clinical_test", "systems": ["system_a", "system_c"], "sample_size": 150, "seed": 42, "n_bootstrap": 100, "format": "markdown"},
        )
        self.assertEqual(exp_md.status_code, 200)
        self.assertIn("ACADEMIC PROTOTYPE ONLY", exp_md.text)

        exp_json = self.client.post(
            "/api/evaluation/export",
            headers=hdr,
            json={"dataset_id": "clinical_test", "systems": ["system_c"], "sample_size": 150, "seed": 42, "n_bootstrap": 100, "format": "json"},
        )
        self.assertEqual(exp_json.status_code, 200)
        parsed = json.loads(exp_json.text)
        self.assertEqual(parsed["dataset"]["evaluated_count"], 150)

    def test_index_html_has_separate_evaluation_section(self):
        html = self.client.get("/").text
        self.assertIn('id="evaluationWorkspace"', html)
        self.assertIn('id="evalAuthForm"', html)
        self.assertIn('id="btnRunEvaluation"', html)


if __name__ == "__main__":
    unittest.main()
