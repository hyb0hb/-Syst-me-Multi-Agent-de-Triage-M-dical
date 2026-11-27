"""Expérimentation isolée Agent 3 : entraînement OOF sans fuite & évaluation des stratégies de fusion.

Ne modifie jamais les modèles de production (`agent_decision/model_lr.joblib`,
`agent_clinique/model_rf.joblib`, `agent_image/model_cnn.keras`).
Tous les artefacts expérimentaux sont enregistrés dans `models/experiments/agent3_retraining/`.
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_clinique.train_rf import FEATURES, build_pipeline
from agent_decision.train_decision import make_balanced_quadrant_pairs
from backend.config import PRESETS, SAMPLES_DIR
from backend.services.triage_service import align_clinical_risk

EXP_DIR = ROOT / "models" / "experiments" / "agent3_retraining"
SEED = 42


def git_hash(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        upper = y_prob <= bins[i + 1] if i == n_bins - 1 else y_prob < bins[i + 1]
        mask = (y_prob >= bins[i]) & upper
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_true)) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def triage_tier(p_final: np.ndarray) -> np.ndarray:
    return np.where(p_final >= 0.70, "URGENT", np.where(p_final >= 0.40, "A SURVEILLER", "NON URGENT"))


def evaluate_predictions(y_true: np.ndarray, p_score: np.ndarray, urgent_thresh: float = 0.70) -> dict:
    y_urg = (p_score >= urgent_thresh).astype(int)
    cm = confusion_matrix(y_true, y_urg, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    prob_true, prob_pred = calibration_curve(y_true, np.clip(p_score, 0.0, 1.0), n_bins=10, strategy="uniform")
    tiers = triage_tier(p_score)
    tier_counts = {
        "URGENT": int((tiers == "URGENT").sum()),
        "A_SURVEILLER": int((tiers == "A SURVEILLER").sum()),
        "NON_URGENT": int((tiers == "NON URGENT").sum()),
    }
    urgent_missed_as_non_urgent = int(((y_true == 1) & (p_score < 0.40)).sum())
    urgent_missed_as_monitor = int(((y_true == 1) & (p_score >= 0.40) & (p_score < urgent_thresh)).sum())

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, p_score)),
        "pr_auc": float(average_precision_score(y_true, p_score)),
        "brier_score": float(brier_score_loss(y_true, np.clip(p_score, 0.0, 1.0))),
        "ece": expected_calibration_error(y_true, p_score),
        "accuracy_urgent": float(accuracy_score(y_true, y_urg)),
        "precision_urgent": float(precision_score(y_true, y_urg, zero_division=0)),
        "recall_urgent": float(recall_score(y_true, y_urg, zero_division=0)),
        "f1_urgent": float(f1_score(y_true, y_urg, zero_division=0)),
        "specificity_urgent": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision_non_urgent": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "recall_non_urgent": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "urgent_missed_as_non_urgent": urgent_missed_as_non_urgent,
        "urgent_downgraded_to_surveiller": urgent_missed_as_monitor,
        "tier_counts": tier_counts,
        "calibration_curve": {
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred],
        },
    }


def apply_decision_system(lr_model, pi: np.ndarray, pc: np.ndarray, align_065: bool) -> tuple[np.ndarray, np.ndarray]:
    p_comp = lr_model.predict_proba(np.column_stack([pi, pc]))[:, 1]
    if align_065:
        pc_eff = np.array([align_clinical_risk(float(v), clin_threshold=0.65) for v in pc])
    else:
        pc_eff = pc
    p_fin = np.maximum(p_comp, np.maximum(pi, pc_eff))
    return p_comp, p_fin


def make_realistic_prior_pairs(img_df, clin_df, n_total: int, p_pneu: float, p_stroke: float, rng) -> tuple[np.ndarray, np.ndarray]:
    img_0 = img_df[img_df["label"] == 0]["proba"].values
    img_1 = img_df[img_df["label"] == 1]["proba"].values
    clin_0 = clin_df[clin_df["label"] == 0]["proba"].values
    clin_1 = clin_df[clin_df["label"] == 1]["proba"].values
    y_img = (rng.random(n_total) < p_pneu).astype(int)
    y_clin = (rng.random(n_total) < p_stroke).astype(int)
    pi = np.where(y_img == 1, rng.choice(img_1, n_total, True), rng.choice(img_0, n_total, True))
    pc = np.where(y_clin == 1, rng.choice(clin_1, n_total, True), rng.choice(clin_0, n_total, True))
    y = ((y_img == 1) | (y_clin == 1)).astype(int)
    return np.column_stack([pi, pc]), y


def load_image_predictions() -> pd.DataFrame:
    import tensorflow as tf

    cnn = tf.keras.models.load_model(ROOT / "agent_image" / "model_cnn.keras")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        ROOT / "data" / "chest_xray" / "test",
        labels="inferred",
        label_mode="binary",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
        seed=SEED,
    )
    y_img = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    p_img = cnn.predict(test_ds, verbose=0).ravel()
    return pd.DataFrame({"proba": p_img, "label": y_img})


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any temporary scratch files if present
    for tmp_name in ["_baseline_main_rf.joblib", "_cache_agent1_test_preds.csv",
                     "_cache_agent2_main_test_preds.csv", "_cache_agent2_voie1_test_preds.csv",
                     "_cache_agent2_voie1_oof_train.csv"]:
        tmp_path = EXP_DIR / tmp_name
        if tmp_path.exists():
            tmp_path.unlink()

    main_hash = git_hash("main")
    voie1_hash = git_hash("feature/agent2-voie1-optimisee")
    head_hash = git_hash("HEAD")

    # 1. Load datasets cleanly with identical random_state=42 split
    df = pd.read_csv(ROOT / "data" / "healthcare-dataset-stroke-data.csv")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    X, y = df[FEATURES], df["stroke"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 2. Load production and main-baseline models cleanly (never mutating prod files)
    main_rf_bytes = subprocess.check_output(["git", "show", f"{main_hash}:agent_clinique/model_rf.joblib"], cwd=ROOT)
    rf_main = joblib.load(io.BytesIO(main_rf_bytes))
    rf_voie1 = joblib.load(ROOT / "agent_clinique" / "model_rf.joblib")
    lr_prod = joblib.load(ROOT / "agent_decision" / "model_lr.joblib")
    if not hasattr(lr_prod, "multi_class"):
        lr_prod.multi_class = "auto"

    # 3. Generate strict zero-leakage 5-fold OOF predictions on X_train (4088 rows, 199 stroke)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v1 = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_df = pd.DataFrame({"proba": oof_v1, "label": y_train.values})
    oof_df.to_csv(EXP_DIR / "oof_predictions_agent2_voie1.csv", index=False)

    clin_test_main = pd.DataFrame({"proba": rf_main.predict_proba(X_test)[:, 1], "label": y_test.values})
    clin_test_v1 = pd.DataFrame({"proba": rf_voie1.predict_proba(X_test)[:, 1], "label": y_test.values})

    # 4. Agent 1 predictions (frozen CNN) on chest_xray/test -> strict 50/50 Split A / Split B
    img_test = load_image_predictions()
    img_a, img_b = train_test_split(img_test, test_size=0.5, stratify=img_test["label"], random_state=SEED)

    # 5. Train experimental Agent 3 (OOF Logistic Regression) strictly on img_a x OOF Voie 1 (ZERO X_test, ZERO img_b)
    rng_tr = np.random.default_rng(SEED)
    X_tr_lr, y_tr_lr = make_balanced_quadrant_pairs(img_a, oof_df, 2000, rng_tr)
    lr_oof = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=SEED)
    lr_oof.fit(X_tr_lr, y_tr_lr)
    joblib.dump(lr_oof, EXP_DIR / "model_lr_oof.joblib")

    # 6. Evaluate on REAL untouched independent cohorts (no synthetic pairing!)
    # 6A. Clinical test set (N=1022, 50 stroke), low-risk normal X-ray context (P_img = 0.0033)
    p_img_normal_ctx = np.full(len(y_test), 0.0033)
    systems_real_clin = {}
    for sys_id, c_df, lr_m, al in [
        ("System_A_Baseline", clin_test_main, lr_prod, False),
        ("System_B_Voie1_OldLR", clin_test_v1, lr_prod, False),
        ("System_B_Align_OldLR", clin_test_v1, lr_prod, True),
        ("System_C1_Voie1_OOFLR_Unaligned", clin_test_v1, lr_oof, False),
        ("System_C2_Voie1_OOFLR_Aligned", clin_test_v1, lr_oof, True),
    ]:
        pc_vec = c_df["proba"].values
        p_comp, p_fin = apply_decision_system(lr_m, p_img_normal_ctx, pc_vec, align_065=al)
        systems_real_clin[sys_id] = evaluate_predictions(y_test.values, p_fin, urgent_thresh=0.70)

    # 6B. Held-out Chest X-Ray img_b (N=312, 195 pneumonia), routine low-risk clinical context (P_clin = 0.0059)
    p_clin_routine_ctx = np.full(len(img_b), 0.0059)
    systems_real_img = {}
    for sys_id, lr_m, al in [
        ("Prod_LR_Unaligned", lr_prod, False),
        ("OOF_LR_Aligned", lr_oof, True),
    ]:
        _, p_fin = apply_decision_system(lr_m, img_b["proba"].values, p_clin_routine_ctx, align_065=al)
        systems_real_img[sys_id] = evaluate_predictions(img_b["label"].values, p_fin, urgent_thresh=0.70)

    # 7. Simulated Multi-Agent Benchmarks on untouched img_b x complete untouched X_test (zero leakage)
    X_quad_m, y_quad_m = make_balanced_quadrant_pairs(img_b, clin_test_main, 1000, np.random.default_rng(SEED))
    X_quad_v, y_quad_v = make_balanced_quadrant_pairs(img_b, clin_test_v1, 1000, np.random.default_rng(SEED))
    X_real_m, y_real_m = make_realistic_prior_pairs(img_b, clin_test_main, 4000, 0.20, 0.05, np.random.default_rng(SEED))
    X_real_v, y_real_v = make_realistic_prior_pairs(img_b, clin_test_v1, 4000, 0.20, 0.05, np.random.default_rng(SEED))

    quad_bench = {}
    real_prior_bench = {}
    eval_rows = []
    for sys_id, Xq, yq, Xr, yr, lr_m, al in [
        ("System_A_Baseline", X_quad_m, y_quad_m, X_real_m, y_real_m, lr_prod, False),
        ("System_B_Voie1_OldLR", X_quad_v, y_quad_v, X_real_v, y_real_v, lr_prod, False),
        ("System_B_Align_OldLR", X_quad_v, y_quad_v, X_real_v, y_real_v, lr_prod, True),
        ("System_C1_Voie1_OOFLR_Unaligned", X_quad_v, y_quad_v, X_real_v, y_real_v, lr_oof, False),
        ("System_C2_Voie1_OOFLR_Aligned", X_quad_v, y_quad_v, X_real_v, y_real_v, lr_oof, True),
    ]:
        p_comp_q, p_fin_q = apply_decision_system(lr_m, Xq[:, 0], Xq[:, 1], align_065=al)
        p_comp_r, p_fin_r = apply_decision_system(lr_m, Xr[:, 0], Xr[:, 1], align_065=al)
        quad_bench[sys_id] = {
            "p_composite_0.50": evaluate_predictions(yq, p_comp_q, urgent_thresh=0.50),
            "deployed_p_final_0.70": evaluate_predictions(yq, p_fin_q, urgent_thresh=0.70),
        }
        real_prior_bench[sys_id] = {
            "deployed_p_final_0.70": evaluate_predictions(yr, p_fin_r, urgent_thresh=0.70),
        }
        if sys_id in ("System_A_Baseline", "System_C1_Voie1_OOFLR_Unaligned", "System_C2_Voie1_OOFLR_Aligned"):
            for i in range(len(yq)):
                eval_rows.append({
                    "system": sys_id, "y_true": int(yq[i]), "p_img": float(Xq[i, 0]),
                    "p_clin": float(Xq[i, 1]), "p_composite": float(p_comp_q[i]),
                    "p_final": float(p_fin_q[i]), "pred_urgent": int(p_fin_q[i] >= 0.70),
                })
    pd.DataFrame(eval_rows).to_csv(EXP_DIR / "eval_predictions_systems.csv", index=False)

    # 8. 5-Fold Disjoint Image Stability Check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stability = {}
    for cfg_name, use_oof_lr, al in [
        ("System_B_Voie1_OldLR", False, False),
        ("System_C1_Voie1_OOFLR_Unaligned", True, False),
        ("System_C2_Voie1_OOFLR_Aligned", True, True),
    ]:
        aucs, recs, f1s, fns, fps = [], [], [], [], []
        for idx, (f_tr, f_te) in enumerate(skf.split(img_test, img_test["label"])):
            ia_f, ib_f = img_test.iloc[f_tr], img_test.iloc[f_te]
            if use_oof_lr:
                Xf, yf = make_balanced_quadrant_pairs(ia_f, oof_df, 1000, np.random.default_rng(SEED))
                mf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=SEED).fit(Xf, yf)
            else:
                mf = lr_prod
            Xev_f, yev_f = make_balanced_quadrant_pairs(ib_f, clin_test_v1, 1000, np.random.default_rng(100 + idx))
            _, pf = apply_decision_system(mf, Xev_f[:, 0], Xev_f[:, 1], align_065=al)
            tn, fp, fn, tp = confusion_matrix(yev_f, (pf >= 0.70).astype(int)).ravel()
            aucs.append(roc_auc_score(yev_f, pf))
            recs.append(tp / (tp + fn))
            f1s.append(f1_score(yev_f, pf >= 0.70))
            fns.append(fn)
            fps.append(fp)
        stability[cfg_name] = {
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "mean_fn": float(np.mean(fns)), "mean_fp": float(np.mean(fps)),
        }

    # 9. Presets evaluation (+ borderline synthetic stroke case P_clin=0.66)
    import tensorflow as tf
    cnn = tf.keras.models.load_model(ROOT / "agent_image" / "model_cnn.keras")
    preset_results = {}
    for key, p in PRESETS.items():
        img_name = Path(p["sample_image"]).name
        pil_img = tf.keras.utils.load_img(SAMPLES_DIR / img_name, target_size=(224, 224))
        p_img = float(cnn(np.expand_dims(tf.keras.utils.img_to_array(pil_img), 0), training=False).numpy()[0][0])
        row = pd.DataFrame([{
            "age": p["age"], "avg_glucose_level": p["glucose"], "bmi": p["bmi"],
            "hypertension": p["hypertension"], "heart_disease": p["heart_disease"],
            "gender": p["gender"], "ever_married": p["ever_married"],
            "work_type": p["work_type"], "Residence_type": p["residence"], "smoking_status": p["smoking"],
        }])
        pc_main = float(rf_main.predict_proba(row)[0, 1])
        pc_v1 = float(rf_voie1.predict_proba(row)[0, 1])
        entries = {}
        for sname, pc_val, lm, al in [
            ("System_A_Baseline", pc_main, lr_prod, False),
            ("System_B_Voie1_OldLR", pc_v1, lr_prod, False),
            ("System_C1_OOFLR_Unaligned", pc_v1, lr_oof, False),
            ("System_C2_OOFLR_Aligned", pc_v1, lr_oof, True),
        ]:
            pcomp, pfin = apply_decision_system(lm, np.array([p_img]), np.array([pc_val]), align_065=al)
            pf = float(pfin[0])
            entries[sname] = {
                "p_img": round(p_img, 4), "p_clin": round(pc_val, 4),
                "p_composite": round(float(pcomp[0]), 4), "p_final": round(pf, 4),
                "niveau": str(triage_tier(np.array([pf]))[0]),
            }
        preset_results[key] = entries

    # Borderline 0.66 clinical check
    pc_b, pf_un = apply_decision_system(lr_oof, np.array([0.0033]), np.array([0.66]), align_065=False)
    _, pf_al = apply_decision_system(lr_oof, np.array([0.0033]), np.array([0.66]), align_065=True)
    preset_results["boundary_stroke_0.66"] = {
        "unaligned_0.70": {"p_composite": round(float(pc_b[0]), 4), "p_final": round(float(pf_un[0]), 4), "niveau": str(triage_tier(pf_un)[0])},
        "aligned_0.65": {"p_composite": round(float(pc_b[0]), 4), "p_final": round(float(pf_al[0]), 4), "niveau": str(triage_tier(pf_al)[0])},
    }

    training_config = {
        "git": {"main": main_hash, "feature_agent2_voie1_optimisee": voie1_hash, "head": head_hash},
        "seed": SEED,
        "dataset_audit": {
            "cross_patient_paired": False,
            "clinical_train_rows": int(len(X_train)), "clinical_train_positives": int(y_train.sum()),
            "clinical_test_rows": int(len(X_test)), "clinical_test_positives": int(y_test.sum()),
            "image_test_split_a_rows": int(len(img_a)), "image_test_split_a_pneumonia": int(img_a["label"].sum()),
            "image_test_split_b_rows": int(len(img_b)), "image_test_split_b_pneumonia": int(img_b["label"].sum()),
            "legacy_agent3_split_a_clinical_positives": 25,
            "experimental_oof_clinical_positives": int(y_train.sum()),
        },
        "agent3_oof_lr": {
            "estimator": "LogisticRegression",
            "hyperparameters": {"class_weight": "balanced", "C": 1.0, "max_iter": 1000, "random_state": SEED},
            "training_samples": int(len(X_tr_lr)),
            "coefficients": {
                "image": float(lr_oof.coef_[0][0]),
                "clinique": float(lr_oof.coef_[0][1]),
                "intercept": float(lr_oof.intercept_[0]),
            },
            "prod_coefficients": {
                "image": float(lr_prod.coef_[0][0]),
                "clinique": float(lr_prod.coef_[0][1]),
                "intercept": float(lr_prod.intercept_[0]),
            },
        },
    }

    comparison = {
        "real_clinical_test_1022_normal_xray_context": systems_real_clin,
        "real_image_test_312_low_clin_context": systems_real_img,
        "simulated_quadrant_benchmark_4000": quad_bench,
        "simulated_realistic_prior_20p_5p_4000": real_prior_bench,
        "stability_5fold_disjoint_images": stability,
        "presets": preset_results,
    }

    (EXP_DIR / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (EXP_DIR / "comparison_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("[OK] Artefacts experimentaux generes dans models/experiments/agent3_retraining/")


if __name__ == "__main__":
    main()
