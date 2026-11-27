# MedTriage AI — Reproducible Evaluation System Technical Report

> **ACADEMIC PROTOTYPE ONLY — Not a clinically validated medical device. Probabilities and triage categories are experimental research outputs and do not diagnose, confirm, or rule out any disease.**

## 1. Methodology & Anti-Leakage Architecture

1. **Strict Held-Out Separation (`backend/services/evaluation_service.py`):**
   - Clinical dataset (`healthcare-dataset-stroke-data.csv`, 5,110 rows): canonical stratified 80/20 split (`random_state=42`) excludes all `4088` training records and evaluates exclusively on the `1022` held-out patients (`50` strokes, `972` non-strokes, split SHA-256 `79c32bef38677a81`).
   - Radiology dataset (`chest_xray/test`): `624` held-out test X-rays (`390` PNEUMONIA, `234` NORMAL, split SHA-256 `0bdedd95b5b31a81`), plus strict held-out Split B (`312` X-rays, `195` PNEUMONIA, SHA-256 `9d9f3f00d95d7b6a`) never seen during experimental Agent 3 LR fitting.
2. **Label-Blind Inference Firewall:** `_extract_clinical_features` strips `stroke` and `id` before RF/LR inference and preserves missing `bmi` (`NaN`) intact. `_predict_radiology_paths_blind` feeds raw RGB tensors from file paths without exposing class directory names. Ground truth comparison executes strictly post-inference.
3. **Unpaired Cohort Honesty:** Chest X-Ray (pediatric) and Stroke (adult/senior) records come from independent sources (`cross_patient_paired: false`). Clinical triage pathway (`clinical_test`) and radiology pneumonia detection (`radiology_test`, `radiology_test_b`) are evaluated **separately**; cross-dataset quadrant orchestration (`simulated_quadrant`) is isolated and explicitly badged as synthetic simulation.
4. **Statistical Reliability:** Every metric uses **1,000 stratified bootstrap resamples (`seed=42`)** for 95% confidence intervals, plus exact **McNemar tests** and paired bootstrap deltas computed over **identical sample IDs** across System A, System B, and System C.
5. **Authorized Website Evaluation Section:** Separate `Research Evaluation` workspace (`/api/evaluation/metadata`, `/api/evaluation/run`, `/api/evaluation/export`) protected by `X-Eval-Token`, supporting dataset/system selection, seeded random subsampling with reproducible IDs, false-positive/false-negative review, and JSON/Markdown report export.

---

## 2. Independent Clinical Held-Out Evaluation (`clinical_test`, N=1,022, 50 Strokes)

Evaluated blindly on raw clinical features with reference normal chest X-ray (`normal.jpg`) to isolate clinical urgent triage decisions (`>= 0.70` vs `< 0.70`).

### Clinical Triage Pathway Comparison (95% Stratified Bootstrap Confidence Intervals)
| System | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | Brier | ECE | Slope / Intercept | CM `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|---|---|---|---|
| **System A (MAIN)** | 0.8459 [0.782, 0.894] | 0.2412 [0.175, 0.354] | 0.4000 [0.260, 0.540] | 0.9424 [0.927, 0.958] | 0.2632 [0.184, 0.353] | 0.3175 [0.222, 0.422] | 0.1282 | 0.2167 | 0.9267 / -2.4349 | `[[916, 56], [30, 20]]` (FN=30, FP=56) |
| **System B (Voie1+OldLR)** | 0.8512 [0.793, 0.897] | 0.2455 [0.180, 0.367] | 0.3600 [0.240, 0.500] | 0.9506 [0.936, 0.963] | 0.2727 [0.188, 0.371] | 0.3103 [0.210, 0.420] | 0.1429 | 0.2448 | 1.0302 / -2.5666 | `[[924, 48], [32, 18]]` (FN=32, FP=48) |
| **System C (Voie1+OOFLR+Align)** | 0.8512 [0.793, 0.897] | 0.2455 [0.180, 0.367] | 0.6200 [0.480, 0.760] | 0.8992 [0.879, 0.919] | 0.2403 [0.188, 0.297] | 0.3464 [0.273, 0.419] | 0.1480 | 0.2498 | 0.9995 / -2.6462 | `[[874, 98], [19, 31]]` (FN=19, FP=98) |

## 3. Paired Statistical Significance (Identical Sample IDs)

- **`system_c_vs_system_a`:** Meaningful sensitivity gain (+22.1%, McNemar sensitivity p=0.0010); ROC-AUC delta uncertain (95% CI includes 0). (ΔSens=+0.2214 [0.1195, 0.34], ΔAUC=+0.0057 [-0.0063, 0.0181], ΔF1=+0.0307 [-0.0414, 0.1057], McNemar overall p=7e-05, sens p=0.00098)
- **`system_c_vs_system_b`:** Meaningful sensitivity gain (+25.7%, McNemar sensitivity p=0.0002); ROC-AUC delta uncertain (95% CI includes 0). (ΔSens=+0.2574 [0.14, 0.36], ΔAUC=+0.0000 [0.0, 0.0], ΔF1=+0.0357 [-0.0486, 0.1181], McNemar overall p=0.0, sens p=0.00024)
- **`system_b_vs_system_a`:** Uncertain difference on this sample (95% confidence intervals overlap zero). (ΔSens=-0.0360 [-0.16, 0.0805], ΔAUC=+0.0057 [-0.0063, 0.0181], ΔF1=-0.0051 [-0.1014, 0.0897], McNemar overall p=0.48846, sens p=0.75391)

## 4. Verdict: `improvement_demonstrated`
Improvement demonstrated on held-out urgent detection: System C achieves statistically meaningful sensitivity gain (F1=0.346 vs Baseline 0.318), while noting ROC-AUC delta remains modest and datasets remain unpaired.

## 5. Limitations
- Unpaired datasets: clinical stroke cohort and chest X-ray cohort represent distinct patient populations (cross_patient_paired=false).
- Class imbalance: only 50/1022 (4.9%) stroke positives in evaluated split causes wide confidence intervals on sensitivity and precision.
- Classifier probabilities use balanced class weights (class_weight), inflating probability estimates relative to true clinical prevalence (elevated Brier/ECE).
- No true in-vivo clinical pairing is evaluated here; results do not prove diagnostic validity.
### 2.1 Standalone Agent 2 (Random Forest Operating Cutoff: Sys A `@0.50`, Sys B/C `@0.65`)

| System (Agent 2 Standalone Threshold) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | Brier | ECE | CM `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|---|---|---|
| **System A (MAIN)** (@0.50) | 0.8459 [0.782, 0.894] | 0.2412 [0.175, 0.354] | 0.8000 [0.680, 0.900] | 0.7685 [0.741, 0.795] | 0.1509 [0.127, 0.173] | 0.2540 [0.215, 0.289] | 0.1271 | 0.1911 | `[[747, 225], [10, 40]]` |
| **System B (Voie1+OldLR)** (@0.65) | 0.8512 [0.793, 0.897] | 0.2455 [0.180, 0.367] | 0.6200 [0.480, 0.760] | 0.8992 [0.879, 0.919] | 0.2403 [0.188, 0.297] | 0.3464 [0.273, 0.419] | 0.1417 | 0.2264 | `[[874, 98], [19, 31]]` |
| **System C (Voie1+OOFLR+Align)** (@0.65) | 0.8512 [0.793, 0.897] | 0.2455 [0.180, 0.367] | 0.6200 [0.480, 0.760] | 0.8992 [0.879, 0.919] | 0.2403 [0.188, 0.297] | 0.3464 [0.273, 0.419] | 0.1417 | 0.2264 | `[[874, 98], [19, 31]]` |

### 2.2 Clinical Subgroup Reliability Check (Minimum Support: `n >= 50` and `pos >= 5`)

| Subgroup (System C, Clinical Test N=1,022) | n (Positives) | Supported | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| Age >= 60 | 272 (38) | Yes | 0.7592 | 0.8158 | 0.5855 | 0.3735 |
| Age < 60 | 750 (12) | Yes | 0.7244 | 0.0000 | 0.9986 | 0.0000 |
| Hypertension = Yes | 101 (12) | Yes | 0.8165 | 0.8333 | 0.6629 | 0.3846 |
| Hypertension = No | 921 (38) | Yes | 0.8451 | 0.5526 | 0.9230 | 0.3307 |
| Heart Disease = Yes | 55 (11) | Yes | 0.7252 | 0.8182 | 0.5227 | 0.4390 |
| BMI Recorded | 991 (47) | Yes | 0.8585 | 0.6170 | 0.9047 | 0.3494 |
| BMI Missing (NaN) | 31 (3) | **No** | — | — | — | Omitted: insufficient support (n=31, pos=3; requires n>=50 and pos>=5). |
| Gender = Female | 599 (29) | Yes | 0.8635 | 0.7241 | 0.9035 | 0.4000 |
| Gender = Male | 423 (21) | Yes | 0.8325 | 0.4762 | 0.8930 | 0.2703 |

---

## 3. Independent Radiology Held-Out Evaluation (`radiology_test` N=624 & `radiology_test_b` N=312)

Agent 1 (`MobileNetV2 CNN`) is strictly frozen and identical across System A, System B, and System C.

- **CNN Operating Threshold (`0.50`, Full N=624, 390 Pneumonia):** ROC-AUC=`0.9636` `[0.9489, 0.9769]`, PR-AUC=`0.9756` `[0.9625, 0.9861]`, Sensitivity=`0.9872` `[0.9744, 0.9974]`, Specificity=`0.5897` `[0.5256, 0.6538]`, Precision=`0.8004` `[0.7762, 0.8261]`, F1=`0.8840` `[0.8681, 0.9001]`, Brier=`0.1216`, ECE=`0.1416`, CM=`[[138, 96], [5, 385]]` (FN=`5`, FP=`96`).

### Radiology Split B Urgent Pathway Comparison (95% Stratified Bootstrap Confidence Intervals)
| System | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | Brier | ECE | Slope / Intercept | CM `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|---|---|---|---|
| **System A (MAIN)** | 0.9744 [0.956, 0.988] | 0.9850 [0.975, 0.993] | 0.9795 [0.959, 0.995] | 0.6667 [0.581, 0.752] | 0.8304 [0.797, 0.866] | 0.8988 [0.875, 0.921] | 0.1265 | 0.1558 | 1.0166 / -2.6934 | `[[78, 39], [4, 191]]` (FN=4, FP=39) |
| **System B (Voie1+OldLR)** | 0.9744 [0.956, 0.988] | 0.9850 [0.975, 0.993] | 0.9795 [0.959, 0.995] | 0.6667 [0.581, 0.752] | 0.8304 [0.797, 0.866] | 0.8988 [0.875, 0.921] | 0.1265 | 0.1559 | 1.0168 / -2.6939 | `[[78, 39], [4, 191]]` (FN=4, FP=39) |
| **System C (Voie1+OOFLR+Align)** | 0.9744 [0.956, 0.988] | 0.9850 [0.975, 0.993] | 0.9795 [0.959, 0.995] | 0.6667 [0.581, 0.752] | 0.8304 [0.797, 0.866] | 0.8988 [0.875, 0.921] | 0.1265 | 0.1559 | 1.0168 / -2.6939 | `[[78, 39], [4, 191]]` (FN=4, FP=39) |

## 4. Verdict: `no_improvement_demonstrated`
Agent 1 (CNN MobileNetV2) weights were deliberately preserved unchanged across Systems A, B, and C; pneumonia detection performance is therefore identical across all three systems (expected).

## 5. Limitations
- Agent 1 is unchanged across Systems A/B/C; radiology evaluation verifies stability and false-positive rate (66/234 NORMAL >= 0.70 on full test) rather than cross-system improvement.
- Pediatric pneumonia dataset (Kaggle Chest X-Ray) is not paired with adult stroke clinical records.
---

## 4. Synthetic Unpaired Orchestration Benchmark (`simulated_quadrant`, N=4,000, Not Real Patients)

### Simulated Cross-Dataset Quadrants (95% Stratified Bootstrap Confidence Intervals)
| System | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | Brier | ECE | Slope / Intercept | CM `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|---|---|---|---|
| **System A (MAIN)** | 0.8691 [0.858, 0.879] | 0.9545 [0.950, 0.959] | 0.8717 [0.860, 0.883] | 0.6220 [0.590, 0.650] | 0.8737 [0.865, 0.882] | 0.8727 [0.865, 0.880] | 0.1308 | 0.0827 | 0.5745 / -0.1671 | `[[622, 378], [385, 2615]]` (FN=385, FP=378) |
| **System B (Voie1+OldLR)** | 0.8694 [0.858, 0.880] | 0.9546 [0.951, 0.959] | 0.8640 [0.851, 0.875] | 0.6270 [0.596, 0.655] | 0.8742 [0.865, 0.883] | 0.8691 [0.860, 0.877] | 0.1317 | 0.0874 | 0.5913 / -0.22 | `[[627, 373], [408, 2592]]` (FN=408, FP=373) |
| **System C (Voie1+OOFLR+Align)** | 0.8701 [0.859, 0.881] | 0.9546 [0.950, 0.959] | 0.9070 [0.896, 0.917] | 0.6010 [0.572, 0.629] | 0.8721 [0.864, 0.880] | 0.8892 [0.881, 0.896] | 0.1305 | 0.0805 | 0.5991 / -0.2237 | `[[601, 399], [279, 2721]]` (FN=279, FP=399) |

## 3. Paired Statistical Significance (Identical Sample IDs)

- **`system_c_vs_system_a`:** Meaningful sensitivity gain (+3.5%, McNemar sensitivity p=0.0000); ROC-AUC delta uncertain (95% CI includes 0). (ΔSens=+0.0354 [0.0277, 0.0427], ΔAUC=+0.0010 [-0.0006, 0.0027], ΔF1=+0.0165 [0.012, 0.021], McNemar overall p=0.0, sens p=0.0)
- **`system_c_vs_system_b`:** Meaningful sensitivity gain (+4.3%, McNemar sensitivity p=0.0000); ROC-AUC delta uncertain (95% CI includes 0). (ΔSens=+0.0433 [0.035, 0.0517], ΔAUC=+0.0007 [-0.0003, 0.0018], ΔF1=+0.0203 [0.0151, 0.0257], McNemar overall p=0.0, sens p=0.0)
- **`system_b_vs_system_a`:** Uncertain difference on this sample (95% confidence intervals overlap zero). (ΔSens=-0.0079 [-0.015, -0.0007], ΔAUC=+0.0003 [-0.0009, 0.0016], ΔF1=-0.0037 [-0.0081, 0.0007], McNemar overall p=0.16491, sens p=0.04685)

## 4. Verdict: `improvement_demonstrated`
Improvement demonstrated on held-out urgent detection: System C achieves statistically meaningful sensitivity gain (F1=0.889 vs Baseline 0.873), while noting ROC-AUC delta remains modest and datasets remain unpaired.

## 5. Limitations
- SYNTHETIC SIMULATION: Cross-dataset quadrant pairs are NOT real multimodal patient records (cross_patient_paired=false).
- Artificial 75% positive prevalence (Q10, Q01, Q11) exaggerates F1 and precision relative to clinical prevalence.
---

## 5. Final Verdict & Unresolved Issues

- **Verdict (`clinical_test`):** **`improvement_demonstrated`** — Improvement demonstrated on held-out urgent detection: System C achieves statistically meaningful sensitivity gain (F1=0.346 vs Baseline 0.318), while noting ROC-AUC delta remains modest and datasets remain unpaired.
- **Verdict (`radiology_test`):** **`no_improvement_demonstrated`** — Agent 1 (CNN MobileNetV2) weights were deliberately preserved unchanged across Systems A, B, and C; pneumonia detection performance is therefore identical across all three systems (expected).
- **Key Distinction:** Sensitivity improvement of System C over System A (`+0.2214` [0.1195, 0.34], `McNemar sensitivity p=0.00098`) and over System B (`+0.2574` [0.14, 0.36], `McNemar sensitivity p=0.00024`) is statistically meaningful on the 1,022 held-out patients, whereas overall ROC-AUC delta (`+0.0057`, 95% CI `[-0.0063, 0.0181]`) remains statistically uncertain on 50 positive stroke events.
- **Unresolved Issues:**
  1. No genuinely paired multimodal patient dataset exists to validate true cross-modality pneumonia + stroke co-occurrence.
  2. Balanced training weights raise raw Brier/ECE relative to real prevalence.
  3. Agent 1 CNN produces 28.2% false positives (`66/234` at `>= 0.70`) on normal radiographs.
