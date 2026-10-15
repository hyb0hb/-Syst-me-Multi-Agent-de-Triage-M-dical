
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMG_SIZE = (224, 224)
SEED = 42


def load_ds(directory, batch_size, shuffle, subset=None, validation_split=None):
    """Charge un dossier d'images. Classes (ordre alphabétique) : NORMAL=0, PNEUMONIA=1."""
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
        validation_split=validation_split,
        subset=subset,
    )


def count_images(directory):
    return sum(1 for f in Path(directory).iterdir() if f.is_file())


def build_model():
    """MobileNetV2 pré-entraîné sur ImageNet + tête de classification binaire."""
    # Augmentation légère. Pas de flip horizontal : le coeur est à gauche sur une radio.
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # phase 1 : on gèle le réseau pré-entraîné

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # normalisation attendue par MobileNetV2
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall")],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "chest_xray"))
    parser.add_argument("--epochs", type=int, default=5, help="phase 1 (base gelée)")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="phase 2 (fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir, test_dir = data_dir / "train", data_dir / "test"

    # Le dossier "val" de Kaggle ne contient que 16 images : on prend 20 % du train à la place.
    train_ds = load_ds(train_dir, args.batch_size, True, "training", 0.2)
    val_ds = load_ds(train_dir, args.batch_size, True, "validation", 0.2)
    test_ds = load_ds(test_dir, args.batch_size, False)

    # Le dataset est déséquilibré (plus de pneumonies) -> poids de classes
    n_normal = count_images(train_dir / "NORMAL")
    n_pneu = count_images(train_dir / "PNEUMONIA")
    total = n_normal + n_pneu
    class_weight = {0: total / (2 * n_normal), 1: total / (2 * n_pneu)}
    print(f"Train : {n_normal} NORMAL / {n_pneu} PNEUMONIA -> poids {class_weight}")

    model, base = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        )
    ]

    # Phase 1 : on entraîne uniquement la nouvelle tête
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight, callbacks=callbacks)

    # Phase 2 : fine-tuning des 30 dernières couches de MobileNetV2, petit learning rate
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                  class_weight=class_weight, callbacks=callbacks)

    # Évaluation sur le jeu de test (ordre fixe car shuffle=False)
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    proba = model.predict(test_ds, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
    print("Matrice de confusion :\n", confusion_matrix(y_true, y_pred))
    auc = roc_auc_score(y_true, proba)
    print(f"ROC-AUC test : {auc:.3f}")

    model.save(HERE / "model_cnn.keras")
    pd.DataFrame({"proba": proba, "label": y_true}).to_csv(HERE / "preds_test.csv", index=False)
    report = classification_report(y_true, y_pred, output_dict=True,
                                   target_names=["NORMAL", "PNEUMONIA"])
    with open(HERE / "metrics.json", "w") as f:
        json.dump({"roc_auc": auc, "report": report}, f, indent=2)
    print("Sauvegardé : model_cnn.keras, preds_test.csv, metrics.json")


if __name__ == "__main__":
    main()
