# Audit Complet, Stratégie Agent 3 & Comparaison Expérimentale

> **Avertissement déontologique et méthodologique :** Ce système est un **prototype académique expérimental**. Aucune métrique issue de ce rapport ne constitue une validation clinique *in vivo*. Les datasets d'imagerie thoracique et de risque neuro-vasculaire proviennent de cohortes indépendantes non appariées (`cross_patient_paired: false`).

---

## 1. État du dépôt et empreintes Git (Isolation stricte)

Toutes les vérifications, modifications et expérimentations ont été effectuées exclusivement sur la branche isolée **`experiment/retrain-agent3-new-agent2`**, créée à partir de `feature/agent2-voie1-optimisee`.

| Référence Git | Empreinte (Commit Hash) | État / Rôle |
|---|---|---|
| `main` (et `origin/main`) | `f78b5e626b8e270cd025b91acc3f9eea93b29f1a` | **Intacte** — Baseline initiale (`Old RF` + `Prod LR`) |
| `feature/agent2-voie1-optimisee` (et `origin/...`) | `451ae2bdd3617d97885edc6efcae2819b2732f49` | **Intacte** — Agent 2 Voie 1 validé (`threshold = 0.65`) |
| `experiment/retrain-agent3-new-agent2` | Créée depuis `451ae2b` | **Branche active d'expérimentation** |

Les artefacts de production (`agent_image/model_cnn.keras`, `agent_image/metrics.json`, `agent_clinique/model_rf.joblib`, `agent_clinique/metrics.json`, `agent_decision/model_lr.joblib`, `agent_decision/metrics.json`) sont **strictement préservés à l'identique** (`model_lr.joblib` SHA-256 préfixe `bd10359c545a89c7`, identique à `main`).

---

## 2. Architecture réelle du système

Le flux d'inférence complet tracé dans le dépôt est le suivant :

1. **Agent 1 — Vision thoracique (`agent_image/`) :**
   - Modèle : `MobileNetV2` pré-entraîné ImageNet, fine-tuné en 2 phases (5 époques tête gelée `lr=1e-3`, puis 5 époques sur les 30 dernières couches `lr=1e-5`) avec augmentation légère (rotation/zoom/contrast, sans flip horizontal).
   - Dataset : Kaggle Chest X-Ray (`data/chest_xray/`), `5 216` images train (`1 341 NORMAL`, `3 875 PNEUMONIA` avec `validation_split=0.2` car `val/` Kaggle ne contient que 16 images) et `624` images test (`234 NORMAL`, `390 PNEUMONIA`).
   - Métriques certifiées (`agent_image/metrics.json`) : `ROC-AUC = 0.9647`, `Accuracy@0.50 = 0.8590`, rappel pneumonie `0.9846` (`384/390`), rappel normal `0.6496` (`152/234`). Vérifié directement sur les 624 images test : `ROC-AUC = 0.964727`, `PR-AUC = 0.973583`, `Brier = 0.112678`.
2. **Agent 2 — Données cliniques tabulaires (`agent_clinique/`, Voie 1) :**
   - Pipeline scikit-learn encapsulé (`ColumnTransformer` + `RandomForestClassifier`) sans fuite : `MissingIndicator` sur `bmi`, `SimpleImputer(strategy="median")` sur `avg_glucose_level` & `bmi`, interactions ciblées (`PolynomialFeatures(degree=2, interaction_only=True)` sur `["age", "hypertension", "heart_disease"]`), `OneHotEncoder(handle_unknown="ignore")`, puis 700 arbres régularisés (`max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10`, `max_features=0.5`, `criterion="log_loss"`, `class_weight="balanced_subsample"`).
   - Dataset : Kaggle Stroke (`data/healthcare-dataset-stroke-data.csv`, `5 110` patients, `249` AVC), découpé en `X_train` (`4 088` patients, **`199` AVC**) et `X_test` (`1 022` patients, **`50` AVC**, `972` non-AVC).
   - Seuil OOF gelé (`5-fold StratifiedKFold` sur `X_train` uniquement) : **`threshold = 0.65`** (`oof_validation.roc_auc = 0.8452`, `best_f1 = 0.2805`).
   - Métriques test (`1 022` patients) : `ROC-AUC = 0.8512` (vs `0.8459` sur `main`), `PR-AUC = 0.2455` (vs `0.2412`), `Brier = 0.1417`, `Precision@0.65 = 0.2403`, `Recall@0.65 = 0.6200` (`31/50`), `F1@0.65 = 0.3464`.
3. **Agent 3 & Orchestrateur (`agent_decision/`, `backend/`, `static/app.js`, `demo.py`) :**
   - Modèle : `LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)` prenant en entrée `(P_img, P_clin)`.
   - Poids de production (`agent_decision/metrics.json`) : `image = +3.1397`, `clinique = +3.6108`, `intercept = -2.9170`.
   - Backend (`backend/services/triage_service.py`, `model_service.py`, `config.py`, `schemas/triage.py`), Frontend (`static/app.js`) et CLI (`demo.py`) lisent dynamiquement les seuils/métriques (`threshold = 0.65` pour le statut Agent 2) et appliquent les paliers SAU fixes (`URGENT >= 0.70`, `À SURVEILLER >= 0.40`, `NON URGENT < 0.40`).

---

## 3. Procédure d'entraînement existante d'Agent 3 (`agent_decision/train_decision.py`)

Dans la version héritée (`c1dd9fe` / `f78b5e6`), `agent_decision/train_decision.py` :
1. Lit `--img agent_image/preds_test.csv` et `--clin agent_clinique/preds_test.csv` (**issus des jeux de test**, sachant qu'`agent_image/preds_test.csv` était même absent du disque local).
2. Découpe chacun en deux moitiés stratifiées `50/50` (`img_a` / `img_b` et `clin_a` / `clin_b`).
3. Appelle `make_balanced_quadrant_pairs(..., replace=True)` pour tirer `1 000` paires synthétiques par quadrant (`Q00`, `Q10`, `Q01`, `Q11` = `4 000` paires d'entraînement) sur `Split A`, et `500` par quadrant (`2 000` paires de test) sur `Split B`.
4. Entraîne la `LogisticRegression(class_weight="balanced")` et évalue d'abord `p_composite >= 0.50` (`ROC-AUC = 0.8387`) puis la règle déployée `p_final = max(p_composite, p_img, p_clin) >= 0.70` (`ROC-AUC = 0.8706`).

---

## 4. Faiblesses techniques et scientifiques identifiées

### Problème A — Partitionnement du jeu de test et effectif positif très réduit
- `Split A` et `Split B` provenaient tous deux de `preds_test.csv` (`X_test`, `1 022` patients).
- Décompte réel vérifié dans `clin_a` et `clin_b` : `511` lignes chacun dont seulement **`25` vrais cas d'AVC** pour l'entraînement d'Agent 3 et **`25` vrais cas d'AVC** pour l'évaluation.
- `make_balanced_quadrant_pairs` rééchantillonnait ces `25` vrais AVC avec remise jusqu'à `2 000` paires positives (`Q01 + Q11`) à l'entraînement (~80× en moyenne) et `1 000` au test (~40×).

### Problème B — Absence de cohorte multimodale appariée
- Les deux datasets (`chest_xray` pédiatrique vs `stroke` adultes/seniors) n'ont **aucun identifiant patient commun** (`cross_patient_paired: false`). `data/nih_metadata.csv` ne contient que des métadonnées radiologiques NIH inutilisées.
- Agent 3 ne réalise donc **pas** une fusion biologique intra-patient, mais une **simulation d'orchestration hiérarchique par quadrants**.

### Problème C — Absence de prédictions OOF pour Agent 1 (CNN)
- `agent_image/train_cnn.py` n'a entraîné qu'un seul modèle `model_cnn.keras` (`validation_split=0.2`). Générer des prédictions OOF pour Agent 1 nécessiterait de réentraîner 5 CNNs MobileNetV2 sur les 5 plis de `5 216` images, ce qui modifierait de fait le protocole sans correspondre aux poids figés de `model_cnn.keras`.
- En outre, sur `chest_xray/test` (`234` clichés normaux), le CNN produit `66/234` (**`28.2 %`**) de faux positifs dès `P_img >= 0.70` (malgré un rappel élevé de `379/390 = 97.2 %`), ce qui domine mécaniquement les fausses alertes globales du système lorsque la prévalence de pneumonie est modérée.

### Problème D — Incohérence critique de la règle `p_final = max(p_composite, p_img, p_clin)` avec le seuil `0.65` de Voie 1
- Avec les poids de production (`3.14`, `3.61`, `-2.92`), un patient atteint d'AVC isolé (`P_img = 0.0033`, `P_clin = 0.65`) obtient `p_composite = 0.365` ; `max(...)` vaut donc `0.65 < 0.70` -> **`À SURVEILLER` au lieu d'`URGENT`**, contredisant Agent 2 qui affiche en rouge `"Risque neuro-vasculaire élevé"`.
- Conséquence mesurée sur les **`1 022` patients réels de `X_test`** (contexte radio saine `0.0033`) :
  - **Agent 2 Voie 1 seul (`>= 0.65`)** détecte **`31 / 50` AVC (`62.0 %`)** (`F1 = 0.3464`).
  - Intégré dans l'ancien orchestrateur **`max >= 0.70` (`System B`)**, le rappel chute à **`18 / 50` (`36.0 %`, `F1 = 0.3103`)** — soit **`13 / 50` vrais AVC (`[0.65, 0.70)`) rétrogradés**, faisant même reculer le rappel urgent par rapport à l'ancien Agent 2 (`20 / 50 = 40.0 %` sous `System A`) !
- De plus, `triage_service.py` contenait un commentaire obsolète (`P_clin >= 0.50 -> URGENT`) alors que le code exigeait `0.70`, et `agent_decision/metrics.json` expose en racine `roc_auc = 0.8387` (logit seul) plutôt que `0.8706` (`p_final`).

---

## 5. Analyse de compatibilité des datasets & faisabilité OOF

- **Agent 2 (Faisable à 100 % sans fuite) :** La génération OOF 5-fold stratifiée avec `build_pipeline()` sur `X_train` (`4 088` patients, **`199` vrais AVC**) ajuste l'imputation, `MissingIndicator`, les interactions et l'encodage strictement par pli d'entraînement. `X_test` (`1 022` patients, **`50` vrais AVC**) reste **100 % vierge**.
- **Agent 1 (Faisabilité partielle sans réentraînement) :** Le CNN `model_cnn.keras` reste strictement figé ; ses prédictions sur `chest_xray/test` (`624` clichés) sont partitionnées en `img_a` (`312`, dont `195` pneumonies) pour calibrer Agent 3 et `img_b` (`312`, dont `195` pneumonies) pour l'évaluation étanche finale.
- **Évaluation sans faux appariement :**
  1. Les métriques **sur données patient réelles** sont rapportées **séparément par cohorte** (`1 022` patients cliniques + `312` clichés thoraciques `img_b`).
  2. Le comportement conjoint multi-agent est évalué sur **deux bancs simulés explicites** construits uniquement à partir de `img_b` × `X_test` complet (`50` AVC réels) : grille équilibrée 4 quadrants (`N = 4 000`) et priorité réaliste SAU (`20 % pneumonie / 5 % AVC`, `N = 4 000`), complétés par une **validation croisée par 5 plis disjoints d'images**.

---

## 6. Comparaison des stratégies candidates (A, B, C, D)

Chaque approche a été évaluée avant implémentation :

- **Stratégie A — Conserver Agent 3 inchangé :** simple et compatible, mais laisse la fuite `preds_test.csv` (`25` AVC) et perd `13/50` vrais AVC dans `[0.65, 0.70)`.
- **Stratégie B — Réentraîner uniquement la LR sur OOF (`max >= 0.70` inchangé) :** corrige la fuite clinique (`199` AVC OOF), mais dégrade le F1 d'orchestration (`0.866 -> 0.862`) et garde le rappel AVC bloqué à `36.0 %`.
- **Stratégie C — Améliorer l'architecture de fusion (alignement monotone de seuil SAU `0.65 -> 0.70`) :** des variantes non linéaires (Noisy-OR, features `max`) ont aussi été testées ; le Noisy-OR sans calibration préalable explose les faux positifs sous prévalence réaliste (`FP : 1 037 -> 1 253`), tandis que l'alignement monotone `align_clinical_risk` supprime la contradiction `[0.65, 0.70)` sans toucher aux paliers SAU ni casser la formule de consensus.
- **Stratégie D — Rendre rigoureuse l'orchestration par quadrants (sans faux appariement patient) :** séparation stricte entre métriques par cohorte réelle (`1 022` cliniques + `312` radios) et bancs d'orchestration simulés sur `img_b × X_test` vierge + validation croisée sur 5 plis d'images.

**Décision retenue : combinaison Stratégie C + Stratégie D.**

| Critère | Stratégie A : Garder Agent 3 tel quel | Stratégie B : Réentraîner LR sur OOF (`max >= 0.70`) | Stratégie C+D retenue : **LR OOF sans fuite + Alignement de seuil SAU (`0.65 -> 0.70`)** |
|---|---|---|---|
| **Validité scientifique / anti-fuite** | Faible (`Split A` pris sur `preds_test.csv`) | Bonne (`199` AVC OOF, `X_test` intact) | **Optimale** (`199` AVC OOF, `X_test` et `img_b` vierges) |
| **Vrais positifs cliniques (train -> test)** | `25` -> `25` | `199` -> `50` | **`199` -> `50`** |
| **Cohérence seuil Agent 2 (`0.65`) / SAU (`0.70`)** | Cassée (`13/50` AVC perdus dans `[0.65, 0.70)`) | Cassée (`p_comp` encore plus bas ; rappel baisse) | **Résolue** (alignement monotone continu `[0.65,1]->[0.70,1]`) |
| **Rappel AVC réel (`1 022` patients)** | `36.0 %` (`18/50`) | `36.0 %` (`18/50`) | **`62.0 %` (`31/50`)** |
| **F1 / FN sur banc quadrants (5-fold)** | `0.866 ± 0.005` (`452.8` FN) | `0.862 ± 0.004` (`480.6` FN) | **`0.890 ± 0.005` (`304.4` FN)** |
| **Compatibilité UI / API / CLI / Schémas** | Oui (sauf contradiction `[0.65, 0.70)`) | Oui | **100 % compatible** (backend, `app.js`, `demo.py`) |

**Pourquoi Stratégie B seule échoue :** Mathématiquement, une régression logistique linéaire en 2 variables `(P_img, P_clin)` avec `class_weight="balanced"` apprend un hyperplan additif (`w_img=2.7488, w_clin=3.5082, intercept=-2.9053`) qui reste `< 0.50` sur les urgences isolées (`Q01`/`Q10`) afin d'éviter des faux positifs sur les cas sains. Sans aligner la règle d'escalade SAU avec le seuil OOF `0.65` d'Agent 2, `System C1` régresse légèrement par rapport à l'ancien Agent 3.

---

## 7. Changements implémentés (branche isolée uniquement)

1. **`agent_decision/train_experiment_agent3.py` (nouveau script reproductible) :** génère les prédictions OOF 5-fold Voie 1 (`4 088` lignes, `199` AVC), entraîne `model_lr_oof.joblib` sans toucher à `X_test`/`img_b`, et évalue `System A`, `System B`, `System B+Align`, `System C1` et `System C2` dans `models/experiments/agent3_retraining/`.
2. **`backend/services/triage_service.py` :** ajout du helper pur `align_clinical_risk(p_clin, clin_threshold=0.65, urgent_threshold=0.70)` appelé dans `process_decision_agent(...)` avec `clin_threshold` dynamique issu de `METRICS_CACHE["clinique"]["threshold"]` ; correction du commentaire obsolète `P_clin >= 0.50`. Les paliers `_TRIAGE_LEVELS` (`0.40`, `0.70`) et le schéma JSON `TriageResponse` restent inchangés.
3. **`backend/services/model_service.py` :** prise en charge optionnelle de la variable d'environnement `MEDTRIAGE_LR_PATH` permettant de tester/servir `models/experiments/agent3_retraining/model_lr_oof.joblib` sans écraser `agent_decision/model_lr.joblib`.
4. **`demo.py` :** réutilisation partagée de `align_clinical_risk` et du chargement dynamique `_load_clin_threshold()` (et support `MEDTRIAGE_LR_PATH`) pour garantir une parité stricte CLI / FastAPI.
5. **`tests/test_system.py` :** ajout de 5 tests de non-régression couvrant l'alignement `[0.65, 0.70)`, l'élimination du sous-triage à `0.66`, la parité CLI/backend, l'intégrité OOF (`4 088` / `199`) et `MEDTRIAGE_LR_PATH`.

**Artefacts expérimentaux isolés dans `models/experiments/agent3_retraining/` :**
- `model_lr_oof.joblib` (`coef_img=2.7488`, `coef_clin=3.5082`, `intercept=-2.9053`, `seed=42`, `C=1.0`, `class_weight="balanced"`, `N=8 000` paires depuis `img_a × OOF`)
- `training_config.json`
- `oof_predictions_agent2_voie1.csv` (`4 088` lignes)
- `eval_predictions_systems.csv`
- `comparison_metrics.json`

---

## 8. Résultats comparatifs complets

### 8.1 Évaluation sur données patient RÉELLES non appariées

#### A. Cohorte clinique vierge (`X_test`, `N = 1 022` dont `50` AVC réels, contexte radio normale `P_img = 0.0033`)

| Système | ROC-AUC | PR-AUC | Brier | ECE | Précision (Urg.) | Rappel AVC (Urg.) | F1 (Urg.) | Spécificité | Matrice `[[TN, FP], [FN, TP]]` | AVC manqués en `NON URGENT` (`<0.40`) | Répartition `URG / SURV / NON` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Sys A (Baseline `main`)** | 0.8459 | 0.2412 | 0.1279 | 0.2138 | 0.2632 | 0.4000 (`20/50`) | 0.3175 | 0.9424 | `[[916, 56], [30, 20]]` | 9 | `76 / 279 / 667` |
| **Sys B (Voie 1 + Old LR)** | 0.8512 | 0.2455 | 0.1426 | 0.2423 | 0.2727 | 0.3600 (`18/50`) | 0.3103 | 0.9506 | `[[924, 48], [32, 18]]` | **8** | `66 / 359 / 597` |
| **Sys C1 (OOF LR sans align.)** | 0.8512 | 0.2455 | 0.1427 | 0.2424 | 0.2727 | 0.3600 (`18/50`) | 0.3103 | 0.9506 | `[[924, 48], [32, 18]]` | **8** | `66 / 359 / 597` |
| **Sys B+Align (Old LR + 0.65)** | 0.8512 | 0.2455 | 0.1477 | 0.2474 | 0.2403 | **0.6200 (`31/50`)** | **0.3464** | 0.8992 | `[[874, 98], [19, 31]]` | **8** | `129 / 296 / 597` |
| **Sys C2 (OOF LR + 0.65 Align.)** | **0.8512** | **0.2455** | 0.1478 | 0.2476 | 0.2403 | **0.6200 (`31/50`)** | **0.3464** | 0.8992 | `[[874, 98], [19, 31]]` | **8** | `129 / 296 / 597` |

#### B. Cohorte radiologique tenue à l'écart (`img_b`, `N = 312` dont `195` pneumonies, contexte clinique faible `P_clin = 0.0059`)
- **Identique** entre `Prod_LR_Unaligned` et `OOF_LR_Aligned` : `ROC-AUC = 0.9697`, `PR-AUC = 0.9812`, `Brier = 0.1178`, `Accuracy = 0.8718`, `Precision = 0.8414`, `Recall = 0.9795` (`191/195`), `F1 = 0.9052`, `Specificity = 0.6923`, matrice `[[81, 36], [4, 191]]`, pneumonies en `NON URGENT` = `2`. L'alignement `>= 0.65` n'altère aucun cas respiratoire isolé.

### 8.2 Bancs simulés d'orchestration (`img_b` × `X_test` complet, zéro fuite)

#### A. Banc équilibré 4 quadrants (`N = 4 000`, `3 000` urgents / `1 000` sains) — Règle déployée `p_final >= 0.70`

| Système | AUC (`p_final`) | AUC (`p_comp`) | PR-AUC (`p_final`) | Brier | ECE | Acc (`>=0.70`) | Prec (Urg.) | Recall (Urg.) | F1 (Urg.) | Spec (`NON`) | Matrice `[[TN, FP], [FN, TP]]` | Urgents -> `NON` (`<0.40`) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Sys A (Baseline)** | 0.8762 | 0.8466 | 0.9555 | 0.1264 | 0.0834 | 0.8230 | 0.8872 | 0.8753 | 0.8812 | 0.6660 | `[[666, 334], [374, 2626]]` | 118 |
| **Sys B (Voie 1 + Old LR)** | 0.8761 | 0.8455 | 0.9554 | 0.1270 | 0.0854 | 0.8160 | 0.8858 | 0.8663 | 0.8760 | 0.6650 | `[[665, 335], [401, 2599]]` | **105** |
| **Sys C1 (OOF LR sans align.)** | 0.8751 | 0.8455 | 0.9549 | 0.1268 | 0.0868 | 0.8150 | **0.8886** | 0.8613 | 0.8747 | **0.6760** | `[[676, 324], [416, 2584]]` | **105** |
| **Sys B+Align** | **0.8778** | 0.8455 | **0.9560** | 0.1258 | 0.0877 | 0.8453 | 0.8822 | **0.9160** | 0.8988 | 0.6330 | `[[633, 367], [252, 2748]]` | **105** |
| **Sys C2 (OOF LR + 0.65 Align.)** | 0.8768 | 0.8455 | 0.9555 | **0.1256** | 0.0883 | **0.8463** | 0.8843 | 0.9147 | **0.8992** | 0.6410 | `[[641, 359], [256, 2744]]` | **105** |

*(Métriques du logit seul `p_composite @ 0.50` sur ce même banc : `Sys A` Acc `0.8155` / Brier `0.1522` / F1 `0.8732` ; `Sys B` Acc `0.8180` / Brier `0.1478` / F1 `0.8761` ; `Sys C1/C2` Acc `0.7058` / Brier `0.1637` / F1 `0.7751` / Spec `0.7950`.)*

#### B. Simulation à prévalence SAU réaliste (`20 % Pneumonie / 5 % AVC`, `N = 4 000`, `993` urgents réels)

| Système | ROC-AUC | PR-AUC | Brier | ECE | Acc | Prec (Urg.) | Recall (Urg.) | F1 (Urg.) | Spec (`NON`) | `FN` (`nonUrg`) | `FP` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Sys A (Baseline)** | 0.9181 | 0.8555 | **0.2825** | **0.3654** | 0.7208 | 0.4680 | 0.9144 | 0.6192 | 0.6568 | 85 (`27`) | 1 032 |
| **Sys B (Voie 1 + Old LR)** | 0.9178 | 0.8554 | 0.2902 | 0.3776 | 0.7183 | 0.4655 | 0.9094 | 0.6158 | 0.6551 | 90 (`24`) | 1 037 |
| **Sys C1 (OOF LR sans align.)** | 0.9174 | 0.8547 | 0.2867 | 0.3746 | **0.7245** | **0.4715** | 0.9084 | **0.6208** | **0.6638** | 91 (`24`) | **1 011** |
| **Sys C2 (OOF LR + 0.65 Align.)** | **0.9182** | 0.8553 | 0.2892 | 0.3771 | 0.7085 | 0.4574 | **0.9345** | 0.6142 | 0.6339 | **65 (`24`)** | 1 101 |

#### C. Stabilité en validation croisée 5 plis disjoints sur `img_test` × `X_test` (`mean ± std`)

| Système | ROC-AUC | Recall (Urg.) | F1 (Urg.) | `FN` moyens | `FP` moyens |
|---|---|---|---|---|---|
| **Sys B (Voie 1 + Old LR)** | `0.8647 ± 0.0202` | `0.8491 ± 0.0150` | `0.8661 ± 0.0051` | `452.8` | `335.0` |
| **Sys C1 (OOF LR sans align.)** | `0.8639 ± 0.0205` | `0.8398 ± 0.0170` | `0.8621 ± 0.0038` | `480.6` | **`325.4`** |
| **Sys C2 (OOF LR + 0.65 Align.)** | **`0.8655 ± 0.0209`** | **`0.8985 ± 0.0120`** | **`0.8903 ± 0.0052`** | **`304.4`** | `360.4` |

---

## 9. Résultats sur les 4 Presets + Cas frontière `0.66`

| Scénario | `P_img` | `P_clin` (Main -> Voie 1) | `p_comp` (Sys A -> B -> C1 -> C2) | `p_final` (Sys A -> B -> C2) | Décision finale |
|---|---|---|---|---|---|
| `cas_pneumonie` | `0.9841` | `0.5431` -> `0.4638` | `0.8942` -> `0.8638` -> `0.8064` -> `0.8064` | `0.9841` -> `0.9841` -> `0.9841` | **`URGENT`** |
| `cas_avc` | `0.0033` | `0.7294` -> `0.7233` | `0.4322` -> `0.4267` -> `0.4112` -> `0.4112` | `0.7294` -> `0.7233` -> **`0.7628`** | **`URGENT`** |
| `cas_critique_mixte` | `0.9841` | `0.8060` -> `0.7981` | `0.9562` -> `0.9550` -> `0.9308` -> `0.9308` | `0.9841` -> `0.9841` -> `0.9841` | **`URGENT`** |
| `cas_routine` | `0.0033` | `0.0002` -> `0.0059` | `0.0519` -> `0.0529` -> `0.0534` -> `0.0534` | `0.0519` -> `0.0529` -> `0.0534` | **`NON URGENT`** |
| *Cas frontière* `(0.0033, 0.66)` | `0.0033` | `0.6600` (Seuil élevé Voie 1) | `0.3587` | `0.6600` (`À SURVEILLER`) -> **`0.7086`** | **`URGENT`** |

---

## 10. Tests automatisés

Commande exécutée après toutes les modifications :
`python -m unittest tests/test_system.py -v`

- **Résultat :** **`Ran 23 tests in 23.704s — OK`** (**23 passés, 0 échec, 0 ignoré**).
- Vérifie les 18 tests existants (validation 422, presets, `/api/metrics`, `MissingIndicator` + NaN batch/single invariance) **et** les 5 nouveaux tests (monotonie de `align_clinical_risk`, résolution du sous-triage à `0.66`, parité `demo.py` / `triage_service.py`, intégrité des OOF `4 088 / 199` sans fuite, et chargement optionnel `MEDTRIAGE_LR_PATH`).

---

## 11. Limites identifiées

1. **Absence de patients appariés :** toute évaluation croisée `(P_img, P_clin)` reste une simulation statistique tant qu'une cohorte réelle comportant radio thoracique + dossier clinique complet des mêmes patients n'est pas disponible.
2. **Brier/ECE sous prévalence réaliste (`0.289` / `0.377`) :** les agents 1 et 2 ayant été entraînés avec pondération de classes inverses (`class_weight`), leurs probabilités brutes sont décalées vers le haut par rapport aux prévalences réelles ; en environnement clinique réel, les faux positifs viennent majoritairement d'Agent 1 (`66/234 = 28.2 %` des radios normales scorées `>= 0.70`).
3. **Absence d'OOF 5-fold pour Agent 1 :** le CNN reste évalué via une partition `img_a / img_b` de `chest_xray/test` sans réentraînement 5-fold.

---

## 12. Recommandation finale

### **IMPROVE THE FUSION ARCHITECTURE**

- **Réentraîner simplement la Régression Logistique sur les mêmes règles (`System C1`) ne suffit pas** : ses performances réelles stagnent (`36.0 %` de rappel AVC) et ses métriques d'orchestration diminuent légèrement (`F1 0.866 -> 0.862` en 5-fold).
- **L'amélioration décisive provient de l'architecture de fusion** :
  1. Supprimer la fuite de `preds_test.csv` en entraînant Agent 3 sur les **199 vrais AVC OOF (`4 088` lignes)** ;
  2. **Synchroniser le principe de précaution SAU sur le seuil OOF certifié d'Agent 2 (`0.65 -> 0.70`)** via `align_clinical_risk` (implémenté proprement dans `triage_service.py` et `demo.py`), ce qui fait bondir la sensibilité aux AVC réels de **`36.0 %` à `62.0 %` (`31/50`, `F1 0.310 -> 0.346`)**, réduit les faux négatifs de **32.8 %** en validation croisée 5 plis (`452.8 -> 304.4`), tout en maintenant les cas sains et les 4 presets cohérents.
- **Note pour déploiement ultérieur :** conserver `model_lr_oof.joblib` avec l'alignement `0.65 -> 0.70` est justifiable car il combine l'entraînement sans fuite (`199` AVC OOF, meilleur Brier `0.1256` et meilleure F1 `0.8992` sur banc quadrants) avec l'escalade clinique cohérente ; aucune fusion vers `main` ne doit cependant être faite sans revue de ce rapport et prise en compte du compromis `Recall (62 % vs 36 %)` vs `FP (+50 sur 972 sains)`.
