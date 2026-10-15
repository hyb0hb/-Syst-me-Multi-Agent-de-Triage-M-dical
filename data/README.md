# Datasets (non versionnés)

Les datasets ne sont pas poussés sur GitHub (trop lourds). Chaque membre les télécharge en local.

## 1. Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Résultat attendu :

```
data/chest_xray/
├── train/NORMAL, train/PNEUMONIA
├── val/NORMAL,   val/PNEUMONIA
└── test/NORMAL,  test/PNEUMONIA
```

> Si l'archive crée un dossier `chest_xray/chest_xray/`, déplacez le contenu d'un niveau
> ou utilisez l'option `--data-dir` du script.

## 2. Stroke Prediction Dataset
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Résultat attendu : `data/healthcare-dataset-stroke-data.csv`

## Téléchargement avec la CLI Kaggle (optionnel)

```bash
pip install kaggle          # + placer kaggle.json dans ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p data --unzip
```
