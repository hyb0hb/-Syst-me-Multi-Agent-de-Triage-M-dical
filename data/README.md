# Datasets — Téléchargement & Problèmes rencontrés

Les datasets ne sont **pas versionnés** sur GitHub (trop lourds). Chaque membre les télécharge en local.

---

## 1. Chest X-Ray Images (Pneumonie) — Agent Image

**Source :** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Structure attendue après téléchargement :**
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 2. Stroke Prediction Dataset — Agent Clinique

**Source :** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

**Fichier attendu :** `data/healthcare-dataset-stroke-data.csv`

---

## Téléchargement avec la CLI Kaggle

```bash
pip install kaggle          # + placer kaggle.json dans ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p data --unzip
```

---

## ⚠️ Problèmes rencontrés & Solutions appliquées

### Problème 1 — Dossier imbriqué après décompression (Chest X-Ray)

**Symptôme :** Après décompression de l'archive Kaggle, le contenu se retrouvait dans
`data/chest_xray/chest_xray/` (un niveau de dossier en trop), ce qui empêchait le script
`train_cnn.py` de trouver les images.

**Solution :** Déplacer le contenu d'un niveau vers le haut :
```bash
# Windows PowerShell
Move-Item data\chest_xray\chest_xray\* data\chest_xray\
```
Ou utiliser l'option `--data-dir` du script :
```bash
python agent_image/train_cnn.py --data-dir data/chest_xray/chest_xray
```

---

### Problème 2 — Jeu de validation Kaggle trop petit (Chest X-Ray)

**Symptôme :** Le dossier `val/` fourni par Kaggle ne contient que **16 images** au total
(8 NORMAL, 8 PNEUMONIA), ce qui rendait la validation non représentative et les métriques
instables pendant l'entraînement.

**Solution :** Le dossier `val/` Kaggle a été ignoré. On utilise à la place **20 % du
dossier `train/`** comme ensemble de validation, grâce au paramètre `validation_split=0.2`
de `keras.utils.image_dataset_from_directory`. Cela donne ~1 048 images de validation
contre 16 auparavant.

```python
# train_cnn.py — ligne clé
train_ds = load_ds(train_dir, batch_size, True, "training",   validation_split=0.2)
val_ds   = load_ds(train_dir, batch_size, True, "validation", validation_split=0.2)
```

---

### Problème 3 — Déséquilibre de classes (Chest X-Ray)

**Symptôme :** Le dataset d'entraînement contient ~74 % de cas PNEUMONIA vs ~26 % de cas
NORMAL. Sans correction, le modèle CNN apprenait à prédire "pneumonie" quasi-systématiquement
(biais vers la classe majoritaire), avec un rappel très faible sur les cas NORMAL.

**Solution :** Calcul automatique de **poids de classes inverses à la fréquence** et passage
à Keras via `class_weight` :

```python
n_normal = count_images(train_dir / "NORMAL")
n_pneu   = count_images(train_dir / "PNEUMONIA")
total    = n_normal + n_pneu
class_weight = {0: total / (2 * n_normal), 1: total / (2 * n_pneu)}
model.fit(..., class_weight=class_weight)
```

---

### Problème 4 — Valeurs manquantes dans la colonne `bmi` (Stroke Dataset)

**Symptôme :** La colonne `bmi` du fichier `healthcare-dataset-stroke-data.csv` contenait
des valeurs `"N/A"` (chaînes de caractères) au lieu de `NaN` numériques. Cela provoquait
une erreur de type lors du prétraitement scikit-learn.

**Solution :** Conversion forcée en numérique avant tout traitement, puis imputation par
la **médiane** via un `SimpleImputer` dans le Pipeline :

```python
# train_rf.py
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")  # "N/A" -> NaN

# Pipeline : imputation médiane automatique
num_pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ...
])
```

---

### Problème 5 — Fort déséquilibre de classes (Stroke Dataset)

**Symptôme :** Seulement **~4,9 %** des patients ont subi un AVC (`stroke=1`).
Le Random Forest prédit systématiquement 0 (pas d'AVC) et obtient une précision globale
de 95 % mais un rappel AVC de 0 % — inutile médicalement.

**Solution :** Utilisation de `class_weight="balanced_subsample"` dans le
`RandomForestClassifier`, qui rééquilibre automatiquement chaque arbre :

```python
rf = RandomForestClassifier(
    class_weight="balanced_subsample",
    ...
)
```
Résultat : **ROC-AUC = 0.846** avec détection effective des cas d'AVC.

---

### Problème 6 — Biais de prévalence dans la fusion des agents (Agent Décision)

**Symptôme :** Le dataset Chest X-Ray contient 62,5 % de pneumonies, tandis que le Stroke
Dataset contient 4,9 % d'AVC. Lors de la fusion naïve des probabilités, le signal
radiologique écrasait le signal clinique d'un **facteur ×6,6**, rendant les urgences AVC
invisibles (un patient avec AVC et radio normale était classé "non urgent").

**Solution :** Méthode d'**échantillonnage par quadrants équilibrés** (`make_balanced_quadrant_pairs`)
qui construit un jeu d'entraînement synthétique représentant équitablement les 4 situations
cliniques :

| Quadrant | Radio | Clinique | Urgence |
|---|---|---|---|
| Q00 | Normale | Pas d'AVC | ❌ Non urgent |
| Q10 | Pneumonie | Pas d'AVC | ✅ Urgent |
| Q01 | Normale | AVC | ✅ Urgent |
| Q11 | Pneumonie | AVC | 🚨 Très urgent |

Résultat : **ROC-AUC de consensus = 0.879**, les deux filières d'urgence sont équitablement
détectées.

---

### Problème 7 — Fuite de données (Data Leakage) dans l'Agent Décision

**Symptôme :** Dans une première version, les prédictions test des agents 1 et 2
(`preds_test.csv`) étaient utilisées **à la fois** pour entraîner et évaluer l'agent 3,
introduisant une fuite qui gonflait artificiellement les métriques.

**Solution :** Découpage strict des `preds_test.csv` en **2 partitions étanches disjointes**
(Split A pour l'apprentissage, Split B pour l'évaluation) avec `train_test_split` stratifié
avant la construction des quadrants :

```python
img_a,  img_b  = train_test_split(img,  test_size=0.5, stratify=img["label"],  random_state=42)
clin_a, clin_b = train_test_split(clin, test_size=0.5, stratify=clin["label"], random_state=42)

X_train, y_train = make_balanced_quadrant_pairs(img_a, clin_a, ...)  # apprentissage
X_test,  y_test  = make_balanced_quadrant_pairs(img_b, clin_b, ...)  # évaluation
```

---

## Résumé des métriques finales après corrections

| Agent | Modèle | Problème principal résolu | ROC-AUC |
|---|---|---|---|
| Agent Image | CNN MobileNetV2 | Déséquilibre classes + val trop petite | ~0.97 |
| Agent Clinique | Random Forest | Valeurs manquantes BMI + déséquilibre AVC | **0.846** |
| Agent Décision | Régression Logistique | Biais de prévalence + data leakage | **0.879** |
