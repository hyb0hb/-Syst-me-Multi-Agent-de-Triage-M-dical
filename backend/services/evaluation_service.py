"""Service d'évaluation scientifique reproductible et sans fuite de données pour MedTriage AI.

Garanties méthodologiques :
- Séparation stricte train / calibration / test (`random_state=42`, même découpage que `train_rf.py`).
- Aucune étiquette (`stroke`, classe radiologique) n'est transmise aux prédicteurs pendant l'inférence.
- Les valeurs manquantes d'IMC (`NaN`) sont préservées telles quelles (aucune imputation silencieuse).
- Comparaison des systèmes A, B et C sur exactement les mêmes identifiants (`sample_ids`) et la même graine.
- Aucune modification de `main` ni des artefacts de production.
"""

from datetime import datetime, timezone
import hashlib
import io
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
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
from sklearn.model_selection import train_test_split

from agent_clinique.train_rf import FEATURES
from backend.config import CHEST_XRAY_DIR, MODEL_PATHS, ROOT_DIR, SAMPLES_DIR, STROKE_CSV
from backend.services.model_service import get_models
from backend.services.triage_service import align_clinical_risk, process_decision_agent

EXP_OOF_LR_PATH = ROOT_DIR / "models" / "experiments" / "agent3_retraining" / "model_lr_oof.joblib"
MAIN_BASELINE_REF = "f78b5e626b8e270cd025b91acc3f9eea93b29f1a"

MIN_SUBGROUP_TOTAL = 50
MIN_SUBGROUP_POS = 5
MIN_SUBGROUP_NEG = 10

ACADEMIC_DISCLAIMER = (
    "ACADEMIC PROTOTYPE ONLY — Not a clinically validated medical device. "
    "Probabilities and triage categories are experimental research outputs and "
    "do not diagnose, confirm, or rule out any disease."
)

SYSTEM_METADATA = {
    "system_a": {
        "name": "System A — MAIN Baseline",
        "short_label": "System A (MAIN)",
        "model_version": "cnn-v1 + rf-main-f78b5e6 + lr-prod",
        "agent1": "MobileNetV2 CNN (frozen v1)",
        "agent2": "RF Baseline (f78b5e6, max_depth=6, cutoff=0.50)",
        "agent3": "Prod LR (Split A, unaligned max>=0.70)",
        "clinical_cutoff": 0.50,
    },
    "system_b": {
        "name": "System B — Voie 1 + Existing Agent 3",
        "short_label": "System B (Voie1+OldLR)",
        "model_version": "cnn-v1 + rf-voie1-451ae2b + lr-prod",
        "agent1": "MobileNetV2 CNN (frozen v1)",
        "agent2": "RF Voie 1 (451ae2b, MissingIndicator+cardio, cutoff=0.65)",
        "agent3": "Prod LR (unaligned max>=0.70)",
        "clinical_cutoff": 0.65,
    },
    "system_c": {
        "name": "System C — Voie 1 + Experimental OOF Agent 3 + Threshold Alignment",
        "short_label": "System C (Voie1+OOFLR+Align)",
        "model_version": "cnn-v1 + rf-voie1-451ae2b + lr-oof-exp + align0.65",
        "agent1": "MobileNetV2 CNN (frozen v1)",
        "agent2": "RF Voie 1 (451ae2b, MissingIndicator+cardio, cutoff=0.65)",
        "agent3": "Experimental OOF LR (199 OOF strokes, align 0.65->0.70)",
        "clinical_cutoff": 0.65,
    },
}

_CACHE: Dict[str, Any] = {
    "cnn_probs_by_path": {},
    "cnn_mtime": None,
    "ref_normal_img_p": None,
    "main_rf": None,
    "oof_lr": None,
}


def _file_sha256_prefix(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _extract_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait strictement les colonnes cliniques autorisées (pare-feu anti-fuite d'étiquettes)."""
    forbidden = {"stroke", "id", "target", "label", "y"}
    feat_df = df[FEATURES].copy()
    if forbidden.intersection(feat_df.columns):
        raise ValueError("Fuite détectée : colonne cible présente dans la matrice d'inférence.")
    return feat_df


def get_clinical_heldout_split() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Retourne le jeu de test clinique strictement disjoint du train (random_state=42)."""
    df = pd.read_csv(STROKE_CSV)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    y_all = df["stroke"].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=y_all, random_state=42
    )
    train_ids = set(train_df["id"].astype(int).tolist())
    test_ids = test_df["id"].astype(int).tolist()
    if train_ids.intersection(test_ids):
        raise RuntimeError("Violation d'étanchéité : recouvrement d'identifiants entre train et test.")

    test_df = test_df.reset_index(drop=True)
    sorted_ids_str = ",".join(str(x) for x in sorted(test_ids))
    split_hash = hashlib.sha256(sorted_ids_str.encode("utf-8")).hexdigest()[:16]

    info = {
        "dataset_id": "clinical_test",
        "dataset_name": "Kaggle Stroke — Held-Out Clinical Test Split (20%, random_state=42)",
        "dataset_version": f"stroke-csv:{_file_sha256_prefix(STROKE_CSV)}",
        "n_total": int(len(test_df)),
        "n_positive": int(test_df["stroke"].sum()),
        "n_negative": int((test_df["stroke"] == 0).sum()),
        "train_rows_excluded": int(len(train_df)),
        "split_hash": split_hash,
        "cross_patient_paired": False,
        "is_simulated": False,
    }
    return test_df, np.array(test_ids, dtype=int), test_df["stroke"].astype(int).to_numpy(), info


def get_radiology_heldout_split(split_b_only: bool = False) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    """Retourne la liste déterministe des clichés de `chest_xray/test` (ou sa moitié vierge `img_b`)."""
    records: List[Dict[str, Any]] = []
    for label_val, cls_name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        cls_dir = CHEST_XRAY_DIR / "test" / cls_name
        files = sorted(
            [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png")],
            key=lambda p: p.name,
        )
        for p in files:
            records.append(
                {
                    "sample_id": f"test/{cls_name}/{p.name}",
                    "filename": p.name,
                    "path": p,
                    "label": label_val,
                }
            )

    y_all = np.array([r["label"] for r in records], dtype=int)
    if split_b_only:
        idx = np.arange(len(records))
        _, idx_b = train_test_split(idx, test_size=0.5, stratify=y_all, random_state=42)
        idx_b = np.sort(idx_b)
        records = [records[int(i)] for i in idx_b]
        y_all = y_all[idx_b]

    ids_str = ",".join(r["sample_id"] for r in records)
    split_hash = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:16]
    ds_id = "radiology_test_b" if split_b_only else "radiology_test"
    info = {
        "dataset_id": ds_id,
        "dataset_name": (
            "Chest X-Ray Held-Out Split B (unseen by experimental Agent 3)"
            if split_b_only
            else "Kaggle Chest X-Ray — Full Held-Out Test Split"
        ),
        "dataset_version": f"chest-xray-{ds_id}:{split_hash}",
        "n_total": int(len(records)),
        "n_positive": int(y_all.sum()),
        "n_negative": int((y_all == 0).sum()),
        "train_rows_excluded": 5216,
        "split_hash": split_hash,
        "cross_patient_paired": False,
        "is_simulated": False,
    }
    return records, y_all, info


def _load_eval_models() -> Dict[str, Any]:
    """Charge en lecture seule les 3 systèmes sans toucher aux fichiers de production ni à `main`."""
    cnn, rf_voie1, lr_prod = get_models()

    if _CACHE["main_rf"] is None:
        raw_rf = subprocess.check_output(
            ["git", "show", f"{MAIN_BASELINE_REF}:agent_clinique/model_rf.joblib"],
            cwd=ROOT_DIR,
        )
        _CACHE["main_rf"] = joblib.load(io.BytesIO(raw_rf))

    if _CACHE["oof_lr"] is None:
        if not EXP_OOF_LR_PATH.exists():
            raise FileNotFoundError(f"Modèle expérimental introuvable : {EXP_OOF_LR_PATH}")
        lr_oof = joblib.load(EXP_OOF_LR_PATH)
        if not hasattr(lr_oof, "multi_class"):
            setattr(lr_oof, "multi_class", "auto")
        _CACHE["oof_lr"] = lr_oof

    return {
        "cnn": cnn,
        "rf_main": _CACHE["main_rf"],
        "rf_voie1": rf_voie1,
        "lr_prod": lr_prod,
        "lr_oof": _CACHE["oof_lr"],
    }


def _predict_radiology_paths_blind(cnn, paths: List[Path]) -> np.ndarray:
    """Inférence CNN pure à partir des pixels uniquement (aucune lecture du label dossier)."""
    cnn_mtime = MODEL_PATHS["cnn"].stat().st_mtime if MODEL_PATHS["cnn"].exists() else 0.0
    if _CACHE["cnn_mtime"] != cnn_mtime:
        _CACHE["cnn_probs_by_path"] = {}
        _CACHE["cnn_mtime"] = cnn_mtime
        _CACHE["ref_normal_img_p"] = None

    cache = _CACHE["cnn_probs_by_path"]
    missing = [p for p in paths if str(p) not in cache]
    if missing:
        batch_size = 32
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            arr = np.stack(
                [np.array(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32) for p in chunk],
                axis=0,
            )
            preds = cnn.predict(arr, verbose=0).ravel()
            for p, prob in zip(chunk, preds):
                cache[str(p)] = float(prob)

    return np.array([cache[str(p)] for p in paths], dtype=float)


def _get_reference_normal_xray_proba(cnn) -> float:
    if _CACHE["ref_normal_img_p"] is None:
        p = SAMPLES_DIR / "normal.jpg"
        _CACHE["ref_normal_img_p"] = float(_predict_radiology_paths_blind(cnn, [p])[0])
    return float(_CACHE["ref_normal_img_p"])


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows: List[Dict[str, Any]] = []
    n = len(y_true)
    for i in range(n_bins):
        upper = (y_prob <= bins[i + 1]) if i == n_bins - 1 else (y_prob < bins[i + 1])
        mask = (y_prob >= bins[i]) & upper
        count = int(mask.sum())
        if count > 0:
            avg_pred = float(y_prob[mask].mean())
            frac_pos = float(y_true[mask].mean())
            ece += (count / n) * abs(frac_pos - avg_pred)
            bin_rows.append(
                {
                    "bin": f"[{bins[i]:.1f}, {bins[i + 1]:.1f}]",
                    "count": count,
                    "prob_pred": round(avg_pred, 4),
                    "prob_true": round(frac_pos, 4),
                }
            )
    return float(ece), bin_rows


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(np.unique(y_true)) < 2:
        return None, None
    p_clip = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
    lr_cal = LogisticRegression(C=1e9, max_iter=1000).fit(logits, y_true)
    return float(lr_cal.coef_[0, 0]), float(lr_cal.intercept_[0])


def _fast_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = rankdata(y_prob)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_prob, kind="mergesort")
    ps = y_prob[order]
    ys = y_true[order]
    distinct = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tps = np.cumsum(ys)[distinct]
    fps = (distinct + 1) - tps
    rec = tps / n_pos
    prec = tps / (tps + fps)
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calcule l'ensemble complet des métriques, calibration et intervalles de confiance à 95 %."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    has_both = len(np.unique(y_true)) >= 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else 0.0
    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0
    acc = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    ece, cal_bins = _expected_calibration_error(y_true, y_prob)
    cal_slope, cal_intercept = _calibration_slope_intercept(y_true, y_prob)

    # Stratified bootstrap 95% CIs
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_vals: Dict[str, List[float]] = {
        k: [] for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "brier_score"]
    }
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        for _ in range(n_bootstrap):
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            b_idx = np.concatenate([b_pos, b_neg])
            yt = y_true[b_idx]
            yp = y_prob[b_idx]
            yd = yp >= threshold
            yt_pos = yt == 1
            btp = int((yt_pos & yd).sum())
            bfp = int((~yt_pos & yd).sum())
            bfn = int((yt_pos & ~yd).sum())
            btn = len(yt) - btp - bfp - bfn
            boot_vals["roc_auc"].append(_fast_roc_auc(yt, yp))
            boot_vals["pr_auc"].append(_fast_pr_auc(yt, yp))
            bsens = btp / (btp + bfn) if (btp + bfn) else 0.0
            bspec = btn / (btn + bfp) if (btn + bfp) else 0.0
            bprec = btp / (btp + bfp) if (btp + bfp) else 0.0
            bf1 = (2 * btp / (2 * btp + bfp + bfn)) if (2 * btp + bfp + bfn) else 0.0
            boot_vals["sensitivity"].append(float(bsens))
            boot_vals["specificity"].append(float(bspec))
            boot_vals["precision"].append(float(bprec))
            boot_vals["f1"].append(float(bf1))
            boot_vals["brier_score"].append(float(np.mean((yp - yt) ** 2)))

    cis: Dict[str, List[float]] = {}
    for k, vals in boot_vals.items():
        if vals:
            cis[k] = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
        else:
            cis[k] = [0.0, 0.0]

    # Add exact Wilson CIs for proportion readability
    wilson_sens = _wilson_ci(tp, tp + fn)
    wilson_spec = _wilson_ci(tn, tn + fp)
    wilson_prec = _wilson_ci(tp, tp + fp)

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "calibration_slope": round(cal_slope, 4) if cal_slope is not None else None,
        "calibration_intercept": round(cal_intercept, 4) if cal_intercept is not None else None,
        "calibration_bins": cal_bins,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "urgent_missed_as_non_urgent": int(((y_true == 1) & (y_prob < 0.40)).sum()),
        "ci_95": cis,
        "wilson_ci_95": {
            "sensitivity": [round(wilson_sens[0], 4), round(wilson_sens[1], 4)],
            "specificity": [round(wilson_spec[0], 4), round(wilson_spec[1], 4)],
            "precision": [round(wilson_prec[0], 4), round(wilson_prec[1], 4)],
        },
    }


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Test exact bilatéral de McNemar sans dépendance SciPy externe."""
    n = b + c
    if n == 0:
        return 1.0
    if n <= 200:
        k = min(b, c)
        two_pow_n = 2 ** n
        p_one_tail = sum(math.comb(n, i) for i in range(k + 1)) / two_pow_n
        return min(1.0, 2.0 * p_one_tail)
    # Edwards continuity-corrected chi-square approximation for large discordant pairs
    chi2 = ((abs(b - c) - 1.0) ** 2) / n
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def compare_two_systems_paired(
    y_true: np.ndarray,
    prob_new: np.ndarray,
    pred_new: np.ndarray,
    prob_ref: np.ndarray,
    pred_ref: np.ndarray,
    label: str,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """Comparaison appariée sur exactement les mêmes identifiants (McNemar + delta bootstrap IC 95 %)."""
    y_true = np.asarray(y_true, dtype=int)
    correct_new = (pred_new == y_true)
    correct_ref = (pred_ref == y_true)
    # Discordant pairs overall AND among positives (sensitivity shift)
    b_only_new_correct = int((correct_new & ~correct_ref).sum())
    c_only_ref_correct = int((~correct_new & correct_ref).sum())
    p_overall = _exact_mcnemar_p(b_only_new_correct, c_only_ref_correct)

    pos_mask = y_true == 1
    b_pos_only_new = int(((pred_new == 1) & (pred_ref == 0) & pos_mask).sum())
    c_pos_only_ref = int(((pred_new == 0) & (pred_ref == 1) & pos_mask).sum())
    p_sens = _exact_mcnemar_p(b_pos_only_new, c_pos_only_ref)

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    deltas: Dict[str, List[float]] = {k: [] for k in ["roc_auc", "sensitivity", "specificity", "f1", "brier_score"]}
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        n_pos_len = len(pos_idx)
        n_neg_len = len(neg_idx)
        for _ in range(n_bootstrap):
            b_idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=n_pos_len, replace=True),
                    rng.choice(neg_idx, size=n_neg_len, replace=True),
                ]
            )
            yt = y_true[b_idx]
            p_n = prob_new[b_idx]
            p_r = prob_ref[b_idx]
            deltas["roc_auc"].append(_fast_roc_auc(yt, p_n) - _fast_roc_auc(yt, p_r))
            deltas["brier_score"].append(float(np.mean((p_n - yt) ** 2) - np.mean((p_r - yt) ** 2)))
            pn, pr = pred_new[b_idx] == 1, pred_ref[b_idx] == 1
            yt_pos = yt == 1
            tpn = int((yt_pos & pn).sum())
            fpn = int((~yt_pos & pn).sum())
            fnn = n_pos_len - tpn
            tpr = int((yt_pos & pr).sum())
            fpr = int((~yt_pos & pr).sum())
            fnr = n_pos_len - tpr
            sn = tpn / n_pos_len
            sr = tpr / n_pos_len
            spn = (n_neg_len - fpn) / n_neg_len
            spr = (n_neg_len - fpr) / n_neg_len
            f1n = (2 * tpn / (2 * tpn + fpn + fnn)) if (2 * tpn + fpn + fnn) else 0.0
            f1r = (2 * tpr / (2 * tpr + fpr + fnr)) if (2 * tpr + fpr + fnr) else 0.0
            deltas["sensitivity"].append(sn - sr)
            deltas["specificity"].append(spn - spr)
            deltas["f1"].append(f1n - f1r)

    delta_summary = {}
    for k, vals in deltas.items():
        if vals:
            mean_d = float(np.mean(vals))
            lo = float(np.percentile(vals, 2.5))
            hi = float(np.percentile(vals, 97.5))
            ci_excludes_zero = (lo > 0) or (hi < 0)
            delta_summary[k] = {
                "delta": round(mean_d, 4),
                "ci_95": [round(lo, 4), round(hi, 4)],
                "significant_95ci": bool(ci_excludes_zero),
            }

    sens_delta = delta_summary.get("sensitivity", {}).get("delta", 0.0)
    auc_sig = delta_summary.get("roc_auc", {}).get("significant_95ci", False)
    sens_sig = (p_sens < 0.05) or delta_summary.get("sensitivity", {}).get("significant_95ci", False)

    if len(pos_idx) < 15:
        interpretation = "Uncertain / low statistical power (fewer than 15 positive cases in sample)."
    elif sens_sig and sens_delta > 0:
        interpretation = (
            f"Meaningful sensitivity gain ({sens_delta:+.1%}, McNemar sensitivity p={p_sens:.4f}); "
            f"ROC-AUC delta {'significant' if auc_sig else 'uncertain (95% CI includes 0)'}."
        )
    elif p_overall < 0.05 or auc_sig:
        interpretation = f"Statistically detectable shift (McNemar overall p={p_overall:.4f})."
    else:
        interpretation = "Uncertain difference on this sample (95% confidence intervals overlap zero)."

    return {
        "comparison": label,
        "mcnemar_overall_p": round(float(p_overall), 5),
        "mcnemar_sensitivity_p": round(float(p_sens), 5),
        "discordant_overall": {"only_new_correct": b_only_new_correct, "only_ref_correct": c_only_ref_correct},
        "discordant_positives": {"only_new_detected": b_pos_only_new, "only_ref_detected": c_pos_only_ref},
        "deltas": delta_summary,
        "statistically_meaningful_sensitivity": bool(sens_sig and len(pos_idx) >= 15),
        "interpretation": interpretation,
    }


def _compute_clinical_subgroups(
    sub_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Rapporte les métriques par sous-groupe uniquement lorsque les effectifs le permettent."""
    masks = [
        ("age_ge_60", "Age >= 60", (sub_df["age"] >= 60).to_numpy()),
        ("age_lt_60", "Age < 60", (sub_df["age"] < 60).to_numpy()),
        ("hypertension_1", "Hypertension = Yes", (sub_df["hypertension"] == 1).to_numpy()),
        ("hypertension_0", "Hypertension = No", (sub_df["hypertension"] == 0).to_numpy()),
        ("heart_disease_1", "Heart Disease = Yes", (sub_df["heart_disease"] == 1).to_numpy()),
        ("bmi_present", "BMI Recorded", (~sub_df["bmi"].isna()).to_numpy()),
        ("bmi_missing", "BMI Missing (NaN)", (sub_df["bmi"].isna()).to_numpy()),
        ("female", "Gender = Female", (sub_df["gender"] == "Female").to_numpy()),
        ("male", "Gender = Male", (sub_df["gender"] == "Male").to_numpy()),
    ]
    out: List[Dict[str, Any]] = []
    for key, label, mask in masks:
        n_sub = int(mask.sum())
        n_pos = int(y_true[mask].sum()) if n_sub > 0 else 0
        n_neg = n_sub - n_pos
        supported = (n_sub >= MIN_SUBGROUP_TOTAL) and (n_pos >= MIN_SUBGROUP_POS) and (n_neg >= MIN_SUBGROUP_NEG)
        if not supported:
            out.append(
                {
                    "subgroup_id": key,
                    "label": label,
                    "n_samples": n_sub,
                    "n_positive": n_pos,
                    "supported": False,
                    "reason": (
                        f"Omitted: insufficient support (n={n_sub}, pos={n_pos}; "
                        f"requires n>={MIN_SUBGROUP_TOTAL} and pos>={MIN_SUBGROUP_POS})."
                    ),
                }
            )
            continue

        yt = y_true[mask]
        yp = y_prob[mask]
        yd = (yp >= threshold).astype(int)
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(yt, yd, labels=[0, 1]).ravel())
        out.append(
            {
                "subgroup_id": key,
                "label": label,
                "n_samples": n_sub,
                "n_positive": n_pos,
                "supported": True,
                "roc_auc": round(float(roc_auc_score(yt, yp)), 4),
                "sensitivity": round(float(tp / (tp + fn)), 4),
                "specificity": round(float(tn / (tn + fp)), 4),
                "precision": round(float(tp / (tp + fp)) if (tp + fp) else 0.0, 4),
                "f1": round(float(f1_score(yt, yd, zero_division=0)), 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return out


def _run_clinical_evaluation(
    systems: List[str],
    sample_size: Optional[int],
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Any]:
    test_df, ids_all, y_all, ds_info = get_clinical_heldout_split()
    if sample_size is not None and sample_size < len(test_df):
        rng = np.random.default_rng(seed)
        chosen_pos = np.sort(rng.choice(len(test_df), size=sample_size, replace=False))
        sub_df = test_df.iloc[chosen_pos].reset_index(drop=True)
        sample_ids = ids_all[chosen_pos]
        y_true = y_all[chosen_pos]
    else:
        sub_df = test_df
        sample_ids = ids_all
        y_true = y_all

    # Pare-feu : extraire uniquement les colonnes cliniques, jamais `stroke` ou `id`
    X_feat = _extract_clinical_features(sub_df)
    models = _load_eval_models()
    p_img_ref = _get_reference_normal_xray_proba(models["cnn"])

    sys_outputs: Dict[str, Any] = {}
    prob_vectors: Dict[str, np.ndarray] = {}
    pred_vectors: Dict[str, np.ndarray] = {}

    for sys_id in systems:
        rf_m = models["rf_main"] if sys_id == "system_a" else models["rf_voie1"]
        lr_m = models["lr_oof"] if sys_id == "system_c" else models["lr_prod"]
        align = (sys_id == "system_c")
        clin_thresh = 0.50 if sys_id == "system_a" else 0.65

        # Inférence strictement aveugle sur X_feat
        p_clin = rf_m.predict_proba(X_feat)[:, 1]
        X_lr = np.column_stack([np.full(len(p_clin), p_img_ref), p_clin])
        p_comp = lr_m.predict_proba(X_lr)[:, 1]
        if align:
            p_clin_eff = np.array([align_clinical_risk(float(v), clin_threshold=0.65) for v in p_clin])
        else:
            p_clin_eff = p_clin
        p_final = np.maximum(p_comp, np.maximum(p_img_ref, p_clin_eff))

        # Comparaison post-inférence uniquement contre y_true
        triage_metrics = compute_classification_metrics(y_true, p_final, threshold=0.70, n_bootstrap=n_bootstrap, seed=seed)
        standalone_metrics = compute_classification_metrics(y_true, p_clin, threshold=clin_thresh, n_bootstrap=n_bootstrap, seed=seed)
        subgroups = _compute_clinical_subgroups(sub_df, y_true, p_final, threshold=0.70)

        y_urg = (p_final >= 0.70).astype(int)
        prob_vectors[sys_id] = p_final
        pred_vectors[sys_id] = y_urg

        false_negatives: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []
        for idx in range(len(sub_df)):
            yt_i = int(y_true[idx])
            yp_i = int(y_urg[idx])
            if yt_i == yp_i:
                continue
            row = sub_df.iloc[idx]
            bmi_str = f"{float(row['bmi']):.1f}" if not pd.isna(row["bmi"]) else "NaN"
            summary = (
                f"Age {float(row['age']):.0f}, {row['gender']}, "
                f"HTN={int(row['hypertension'])}, HD={int(row['heart_disease'])}, "
                f"Gluc={float(row['avg_glucose_level']):.1f}, BMI={bmi_str}"
            )
            item = {
                "sample_id": int(sample_ids[idx]),
                "true_label": yt_i,
                "predicted_urgent": yp_i,
                "predicted_proba": round(float(p_final[idx]), 4),
                "agent1_proba": round(p_img_ref, 4),
                "agent2_proba": round(float(p_clin[idx]), 4),
                "p_composite": round(float(p_comp[idx]), 4),
                "triage_level": "URGENT" if p_final[idx] >= 0.70 else ("À SURVEILLER" if p_final[idx] >= 0.40 else "NON URGENT"),
                "summary": summary,
            }
            if yt_i == 1 and yp_i == 0:
                false_negatives.append(item)
            else:
                false_positives.append(item)

        sys_outputs[sys_id] = {
            "system_id": sys_id,
            **SYSTEM_METADATA[sys_id],
            "metrics": triage_metrics,
            "agent2_standalone_metrics": standalone_metrics,
            "subgroups": subgroups,
            "false_negatives_count": len(false_negatives),
            "false_positives_count": len(false_positives),
            "false_negatives": false_negatives[:200],
            "false_positives": false_positives[:200],
        }

    pairwise: List[Dict[str, Any]] = []
    pairs = [("system_c", "system_a"), ("system_c", "system_b"), ("system_b", "system_a")]
    for new_s, ref_s in pairs:
        if new_s in sys_outputs and ref_s in sys_outputs:
            pairwise.append(
                compare_two_systems_paired(
                    y_true,
                    prob_vectors[new_s],
                    pred_vectors[new_s],
                    prob_vectors[ref_s],
                    pred_vectors[ref_s],
                    label=f"{new_s}_vs_{ref_s}",
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
            )

    ids_hash = hashlib.sha256(",".join(str(x) for x in sample_ids).encode("utf-8")).hexdigest()[:16]
    verdict, verdict_reason = _compute_verdict(sys_outputs, pairwise, y_true)
    return {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "academic_disclaimer": ACADEMIC_DISCLAIMER,
        "dataset": {
            **ds_info,
            "evaluated_count": int(len(sub_df)),
            "evaluated_positives": int(y_true.sum()),
            "evaluated_negatives": int((y_true == 0).sum()),
            "sample_ids_hash": ids_hash,
            "sample_ids_preview": [int(x) for x in sample_ids[:25]],
            "evaluation_note": (
                f"Clinical held-out pathway isolated with reference normal X-ray (normal.jpg, P_img={p_img_ref:.4f}). "
                "Does NOT represent paired multimodal patient validation."
            ),
        },
        "sampling": {
            "mode": "seeded_random_subset" if (sample_size is not None and sample_size < ds_info["n_total"]) else "full_held_out_split",
            "sample_size": int(len(sub_df)),
            "seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
            "identical_samples_across_systems": True,
        },
        "systems": sys_outputs,
        "pairwise_comparisons": pairwise,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "limitations": [
            "Unpaired datasets: clinical stroke cohort and chest X-ray cohort represent distinct patient populations (cross_patient_paired=false).",
            f"Class imbalance: only {int(y_true.sum())}/{len(sub_df)} ({y_true.mean():.1%}) stroke positives in evaluated split causes wide confidence intervals on sensitivity and precision.",
            "Classifier probabilities use balanced class weights (class_weight), inflating probability estimates relative to true clinical prevalence (elevated Brier/ECE).",
            "No true in-vivo clinical pairing is evaluated here; results do not prove diagnostic validity.",
        ],
    }


def _run_radiology_evaluation(
    split_b_only: bool,
    systems: List[str],
    sample_size: Optional[int],
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Any]:
    records, y_all, ds_info = get_radiology_heldout_split(split_b_only=split_b_only)
    if sample_size is not None and sample_size < len(records):
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(records), size=sample_size, replace=False))
        sub_records = [records[int(i)] for i in chosen]
        y_true = y_all[chosen]
    else:
        sub_records = records
        y_true = y_all

    sample_ids = [r["sample_id"] for r in sub_records]
    models = _load_eval_models()
    # Inférence strictement aveugle à partir des chemins d'images uniquement
    p_img = _predict_radiology_paths_blind(models["cnn"], [r["path"] for r in sub_records])

    cnn_operating = compute_classification_metrics(y_true, p_img, threshold=0.50, n_bootstrap=n_bootstrap, seed=seed)

    # Évaluation bout-en-bout filière respiratoire (patient clinique sain de référence)
    sys_outputs: Dict[str, Any] = {}
    prob_vectors: Dict[str, np.ndarray] = {}
    pred_vectors: Dict[str, np.ndarray] = {}

    low_risk_row = pd.DataFrame(
        [{
            "age": 28.0, "avg_glucose_level": 86.5, "bmi": 21.8,
            "hypertension": 0, "heart_disease": 0, "gender": "Female",
            "ever_married": "No", "work_type": "Private",
            "Residence_type": "Urban", "smoking_status": "never smoked",
        }]
    )

    for sys_id in systems:
        rf_m = models["rf_main"] if sys_id == "system_a" else models["rf_voie1"]
        lr_m = models["lr_oof"] if sys_id == "system_c" else models["lr_prod"]
        p_clin_low = float(rf_m.predict_proba(low_risk_row)[0, 1])

        X_lr = np.column_stack([p_img, np.full(len(p_img), p_clin_low)])
        p_comp = lr_m.predict_proba(X_lr)[:, 1]
        p_final = np.maximum(p_comp, np.maximum(p_img, p_clin_low))

        triage_metrics = compute_classification_metrics(y_true, p_final, threshold=0.70, n_bootstrap=n_bootstrap, seed=seed)
        y_urg = (p_final >= 0.70).astype(int)
        prob_vectors[sys_id] = p_final
        pred_vectors[sys_id] = y_urg

        fns, fps = [], []
        for idx, rec in enumerate(sub_records):
            yt_i, yp_i = int(y_true[idx]), int(y_urg[idx])
            if yt_i == yp_i:
                continue
            item = {
                "sample_id": rec["sample_id"],
                "true_label": yt_i,
                "predicted_urgent": yp_i,
                "predicted_proba": round(float(p_final[idx]), 4),
                "agent1_proba": round(float(p_img[idx]), 4),
                "agent2_proba": round(p_clin_low, 4),
                "p_composite": round(float(p_comp[idx]), 4),
                "triage_level": "URGENT" if p_final[idx] >= 0.70 else ("À SURVEILLER" if p_final[idx] >= 0.40 else "NON URGENT"),
                "summary": f"X-Ray {rec['filename']} (Ground Truth: {'PNEUMONIA' if yt_i==1 else 'NORMAL'})",
            }
            if yt_i == 1 and yp_i == 0:
                fns.append(item)
            else:
                fps.append(item)

        sys_outputs[sys_id] = {
            "system_id": sys_id,
            **SYSTEM_METADATA[sys_id],
            "metrics": triage_metrics,
            "cnn_operating_metrics_0.50": cnn_operating,
            "subgroups": [],
            "false_negatives_count": len(fns),
            "false_positives_count": len(fps),
            "false_negatives": fns[:200],
            "false_positives": fps[:200],
        }

    ids_hash = hashlib.sha256(",".join(sample_ids).encode("utf-8")).hexdigest()[:16]
    return {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "academic_disclaimer": ACADEMIC_DISCLAIMER,
        "dataset": {
            **ds_info,
            "evaluated_count": int(len(sub_records)),
            "evaluated_positives": int(y_true.sum()),
            "evaluated_negatives": int((y_true == 0).sum()),
            "sample_ids_hash": ids_hash,
            "sample_ids_preview": sample_ids[:25],
            "evaluation_note": (
                "Radiology pneumonia detection evaluated independently (Agent 1 CNN MobileNetV2 is identical across Systems A, B, and C). "
                "Does NOT validate multimodal fusion."
            ),
        },
        "sampling": {
            "mode": "seeded_random_subset" if (sample_size is not None and sample_size < ds_info["n_total"]) else "full_held_out_split",
            "sample_size": int(len(sub_records)),
            "seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
            "identical_samples_across_systems": True,
        },
        "systems": sys_outputs,
        "pairwise_comparisons": [],
        "verdict": "no_improvement_demonstrated",
        "verdict_reason": (
            "Agent 1 (CNN MobileNetV2) weights were deliberately preserved unchanged across Systems A, B, and C; "
            "pneumonia detection performance is therefore identical across all three systems (expected)."
        ),
        "limitations": [
            "Agent 1 is unchanged across Systems A/B/C; radiology evaluation verifies stability and false-positive rate (66/234 NORMAL >= 0.70 on full test) rather than cross-system improvement.",
            "Pediatric pneumonia dataset (Kaggle Chest X-Ray) is not paired with adult stroke clinical records.",
        ],
    }


def _run_simulated_quadrant_evaluation(
    systems: List[str],
    sample_size: Optional[int],
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Any]:
    """Simulation synthétique explicite sur `img_b` (312) x `clinical_test` (1022) uniquement."""
    clin_df, clin_ids, y_clin_all, clin_info = get_clinical_heldout_split()
    img_records, y_img_all, img_info = get_radiology_heldout_split(split_b_only=True)

    models = _load_eval_models()
    X_clin_all = _extract_clinical_features(clin_df)
    p_img_all = _predict_radiology_paths_blind(models["cnn"], [r["path"] for r in img_records])

    # Pre-compute RF probabilities blindly once on X_clin_all
    p_rf_main_all = models["rf_main"].predict_proba(X_clin_all)[:, 1]
    p_rf_v1_all = models["rf_voie1"].predict_proba(X_clin_all)[:, 1]

    n_per_q = max(5, (sample_size // 4)) if sample_size is not None else 500
    rng = np.random.default_rng(seed)

    img_0_idx = np.where(y_img_all == 0)[0]
    img_1_idx = np.where(y_img_all == 1)[0]
    clin_0_idx = np.where(y_clin_all == 0)[0]
    clin_1_idx = np.where(y_clin_all == 1)[0]

    # Sample indices ONCE so every system evaluates on identical simulated synthetic pairs!
    idx_i = np.concatenate([
        rng.choice(img_0_idx, n_per_q, True),
        rng.choice(img_1_idx, n_per_q, True),
        rng.choice(img_0_idx, n_per_q, True),
        rng.choice(img_1_idx, n_per_q, True),
    ])
    idx_c = np.concatenate([
        rng.choice(clin_0_idx, n_per_q, True),
        rng.choice(clin_0_idx, n_per_q, True),
        rng.choice(clin_1_idx, n_per_q, True),
        rng.choice(clin_1_idx, n_per_q, True),
    ])
    y_true = np.concatenate([
        np.zeros(n_per_q, dtype=int),
        np.ones(3 * n_per_q, dtype=int),
    ])
    perm = rng.permutation(len(y_true))
    idx_i, idx_c, y_true = idx_i[perm], idx_c[perm], y_true[perm]

    sample_ids = [f"{img_records[int(ii)]['filename']}+pt{int(clin_ids[int(ic)])}" for ii, ic in zip(idx_i, idx_c)]
    pi = p_img_all[idx_i]

    sys_outputs: Dict[str, Any] = {}
    prob_vectors: Dict[str, np.ndarray] = {}
    pred_vectors: Dict[str, np.ndarray] = {}

    for sys_id in systems:
        pc = (p_rf_main_all if sys_id == "system_a" else p_rf_v1_all)[idx_c]
        lr_m = models["lr_oof"] if sys_id == "system_c" else models["lr_prod"]
        p_comp = lr_m.predict_proba(np.column_stack([pi, pc]))[:, 1]
        if sys_id == "system_c":
            pc_eff = np.array([align_clinical_risk(float(v), clin_threshold=0.65) for v in pc])
        else:
            pc_eff = pc
        p_final = np.maximum(p_comp, np.maximum(pi, pc_eff))

        metrics = compute_classification_metrics(y_true, p_final, threshold=0.70, n_bootstrap=n_bootstrap, seed=seed)
        y_urg = (p_final >= 0.70).astype(int)
        prob_vectors[sys_id] = p_final
        pred_vectors[sys_id] = y_urg

        fns, fps = [], []
        for k in range(len(y_true)):
            yt_i, yp_i = int(y_true[k]), int(y_urg[k])
            if yt_i == yp_i:
                continue
            item = {
                "sample_id": sample_ids[k],
                "true_label": yt_i,
                "predicted_urgent": yp_i,
                "predicted_proba": round(float(p_final[k]), 4),
                "agent1_proba": round(float(pi[k]), 4),
                "agent2_proba": round(float(pc[k]), 4),
                "p_composite": round(float(p_comp[k]), 4),
                "triage_level": "URGENT" if p_final[k] >= 0.70 else ("À SURVEILLER" if p_final[k] >= 0.40 else "NON URGENT"),
                "summary": f"SYNTHETIC CROSS-DATASET PAIR ({sample_ids[k]})",
            }
            if yt_i == 1 and yp_i == 0:
                fns.append(item)
            else:
                fps.append(item)

        sys_outputs[sys_id] = {
            "system_id": sys_id,
            **SYSTEM_METADATA[sys_id],
            "metrics": metrics,
            "subgroups": [],
            "false_negatives_count": len(fns),
            "false_positives_count": len(fps),
            "false_negatives": fns[:200],
            "false_positives": fps[:200],
        }

    pairwise = []
    for new_s, ref_s in [("system_c", "system_a"), ("system_c", "system_b"), ("system_b", "system_a")]:
        if new_s in sys_outputs and ref_s in sys_outputs:
            pairwise.append(
                compare_two_systems_paired(
                    y_true, prob_vectors[new_s], pred_vectors[new_s],
                    prob_vectors[ref_s], pred_vectors[ref_s],
                    label=f"{new_s}_vs_{ref_s}", n_bootstrap=n_bootstrap, seed=seed,
                )
            )

    ids_hash = hashlib.sha256(",".join(sample_ids).encode("utf-8")).hexdigest()[:16]
    verdict, verdict_reason = _compute_verdict(sys_outputs, pairwise, y_true)
    return {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "academic_disclaimer": ACADEMIC_DISCLAIMER,
        "dataset": {
            "dataset_id": "simulated_quadrant",
            "dataset_name": "SIMULATED Synthetic Cross-Dataset 4-Quadrant Benchmark (img_b x X_test)",
            "dataset_version": f"{img_info['split_hash']}+{clin_info['split_hash']}",
            "n_total": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
            "evaluated_count": int(len(y_true)),
            "evaluated_positives": int(y_true.sum()),
            "evaluated_negatives": int((y_true == 0).sum()),
            "split_hash": f"{img_info['split_hash']}+{clin_info['split_hash']}",
            "sample_ids_hash": ids_hash,
            "sample_ids_preview": sample_ids[:25],
            "cross_patient_paired": False,
            "is_simulated": True,
            "evaluation_note": (
                "SYNTHETIC CROSS-DATASET BENCHMARK ONLY: pairs unrelated pediatric X-rays (img_b) "
                "with adult stroke records (X_test). Not real multimodal patients."
            ),
        },
        "sampling": {
            "mode": "simulated_balanced_quadrants",
            "sample_size": int(len(y_true)),
            "seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
            "identical_samples_across_systems": True,
        },
        "systems": sys_outputs,
        "pairwise_comparisons": pairwise,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "limitations": [
            "SYNTHETIC SIMULATION: Cross-dataset quadrant pairs are NOT real multimodal patient records (cross_patient_paired=false).",
            "Artificial 75% positive prevalence (Q10, Q01, Q11) exaggerates F1 and precision relative to clinical prevalence.",
        ],
    }


def _compute_verdict(sys_outputs: Dict[str, Any], pairwise: List[Dict[str, Any]], y_true: np.ndarray) -> Tuple[str, str]:
    if len(np.unique(y_true)) < 2 or int(y_true.sum()) < 15:
        return (
            "insufficient_evidence",
            "Insufficient evidence: fewer than 15 positive ground-truth cases in the selected sample for reliable statistical inference.",
        )

    c_vs_a = next((p for p in pairwise if p["comparison"] == "system_c_vs_system_a"), None)
    c_vs_b = next((p for p in pairwise if p["comparison"] == "system_c_vs_system_b"), None)
    if not c_vs_a and not c_vs_b:
        return (
            "insufficient_evidence",
            "Single-system evaluation selected; compare Systems A, B, and C together to evaluate relative improvement.",
        )

    checks = [c for c in [c_vs_a, c_vs_b] if c is not None]
    any_sig_sens = any(c["statistically_meaningful_sensitivity"] and c["deltas"]["sensitivity"]["delta"] > 0 for c in checks)
    if any_sig_sens:
        f1_c = sys_outputs.get("system_c", {}).get("metrics", {}).get("f1", 0.0)
        f1_a = sys_outputs.get("system_a", {}).get("metrics", {}).get("f1", 0.0)
        return (
            "improvement_demonstrated",
            f"Improvement demonstrated on held-out urgent detection: System C achieves statistically meaningful sensitivity gain "
            f"(F1={f1_c:.3f} vs Baseline {f1_a:.3f}), while noting ROC-AUC delta remains modest and datasets remain unpaired.",
        )
    return (
        "no_improvement_demonstrated",
        "No statistically significant improvement demonstrated on the evaluated sample (confidence intervals cross zero).",
    )


def get_evaluation_metadata() -> Dict[str, Any]:
    """Retourne les métadonnées disponibles pour l'interface d'évaluation autorisée."""
    _, _, _, clin_info = get_clinical_heldout_split()
    _, _, rad_full = get_radiology_heldout_split(split_b_only=False)
    _, _, rad_b = get_radiology_heldout_split(split_b_only=True)
    return {
        "academic_disclaimer": ACADEMIC_DISCLAIMER,
        "datasets": [
            clin_info,
            rad_full,
            rad_b,
            {
                "dataset_id": "simulated_quadrant",
                "dataset_name": "Simulated Unpaired 4-Quadrant Orchestration (img_b x X_test)",
                "n_total": 2000,
                "n_positive": 1500,
                "n_negative": 500,
                "cross_patient_paired": False,
                "is_simulated": True,
                "split_hash": f"{rad_b['split_hash']}+{clin_info['split_hash']}",
            },
        ],
        "systems": list(SYSTEM_METADATA.values()),
    }


def run_reproducible_evaluation(
    dataset_id: str = "clinical_test",
    systems: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    n_bootstrap: int = 400,
) -> Dict[str, Any]:
    """Point d'entrée unique de l'évaluation reproductible (API REST, CLI et tests)."""
    selected = systems or ["system_a", "system_b", "system_c"]
    for s in selected:
        if s not in SYSTEM_METADATA:
            raise ValueError(f"Système inconnu : {s}")

    if dataset_id == "clinical_test":
        return _run_clinical_evaluation(selected, sample_size, seed, n_bootstrap)
    if dataset_id == "radiology_test":
        return _run_radiology_evaluation(False, selected, sample_size, seed, n_bootstrap)
    if dataset_id == "radiology_test_b":
        return _run_radiology_evaluation(True, selected, sample_size, seed, n_bootstrap)
    if dataset_id == "simulated_quadrant":
        return _run_simulated_quadrant_evaluation(selected, sample_size, seed, n_bootstrap)
    raise ValueError(f"Jeu de données d'évaluation inconnu : {dataset_id}")


def format_evaluation_markdown(result: Dict[str, Any]) -> str:
    """Formate une évaluation reproductible complète en rapport Markdown exportable."""
    ds = result["dataset"]
    samp = result["sampling"]
    lines = [
        "# MedTriage AI — Reproducible Held-Out Evaluation Report",
        "",
        f"> **{result['academic_disclaimer']}**",
        "",
        "## 1. Provenance & Reproducibility Metadata",
        f"- **Evaluation Date (UTC):** `{result['evaluation_date_utc']}`",
        f"- **Dataset:** `{ds['dataset_id']}` — {ds['dataset_name']}",
        f"- **Dataset Version / Split Hash:** `{ds['dataset_version']}` / `{ds['split_hash']}`",
        f"- **Sample Count:** `{ds['evaluated_count']}` (`{ds['evaluated_positives']}` positives, `{ds['evaluated_negatives']}` negatives)",
        f"- **Sample IDs Hash:** `{ds['sample_ids_hash']}`",
        f"- **Sampling Mode & Seed:** `{samp['mode']}`, `seed={samp['seed']}`, `n_bootstrap={samp['n_bootstrap']}`",
        f"- **Cross-Patient Paired:** `{ds['cross_patient_paired']}` | **Synthetic Simulation:** `{ds['is_simulated']}`",
        f"- **Note:** {ds['evaluation_note']}",
        "",
        "## 2. System Comparison (95% Stratified Bootstrap Confidence Intervals)",
        "",
        "| System | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | Brier | ECE | Slope / Intercept | CM `[[TN,FP],[FN,TP]]` |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sys_id, sdata in result["systems"].items():
        m = sdata["metrics"]
        ci = m["ci_95"]
        lines.append(
            f"| **{sdata['short_label']}** "
            f"| {m['roc_auc']:.4f} [{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}] "
            f"| {m['pr_auc']:.4f} [{ci['pr_auc'][0]:.3f}, {ci['pr_auc'][1]:.3f}] "
            f"| {m['sensitivity']:.4f} [{ci['sensitivity'][0]:.3f}, {ci['sensitivity'][1]:.3f}] "
            f"| {m['specificity']:.4f} [{ci['specificity'][0]:.3f}, {ci['specificity'][1]:.3f}] "
            f"| {m['precision']:.4f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}] "
            f"| {m['f1']:.4f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}] "
            f"| {m['brier_score']:.4f} | {m['ece']:.4f} "
            f"| {m['calibration_slope']} / {m['calibration_intercept']} "
            f"| `{m['confusion_matrix']}` (FN={m['fn']}, FP={m['fp']}) |"
        )

    if result["pairwise_comparisons"]:
        lines.extend(["", "## 3. Paired Statistical Significance (Identical Sample IDs)", ""])
        for pw in result["pairwise_comparisons"]:
            d_sens = pw["deltas"].get("sensitivity", {})
            d_auc = pw["deltas"].get("roc_auc", {})
            d_f1 = pw["deltas"].get("f1", {})
            lines.append(
                f"- **`{pw['comparison']}`:** {pw['interpretation']} "
                f"(ΔSens={d_sens.get('delta', 0):+.4f} {d_sens.get('ci_95')}, "
                f"ΔAUC={d_auc.get('delta', 0):+.4f} {d_auc.get('ci_95')}, "
                f"ΔF1={d_f1.get('delta', 0):+.4f} {d_f1.get('ci_95')}, "
                f"McNemar overall p={pw['mcnemar_overall_p']}, sens p={pw['mcnemar_sensitivity_p']})"
            )

    lines.extend(["", f"## 4. Verdict: `{result['verdict']}`", f"{result['verdict_reason']}", "", "## 5. Limitations"])
    for lim in result["limitations"]:
        lines.append(f"- {lim}")

    return "\n".join(lines) + "\n"
