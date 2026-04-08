import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import load_config, build_split_csv, make_dataset
from models.custom_cnn import build_custom_cnn
from models.pretrained import build_pretrained, unfreeze_for_finetuning


def compute_weights(config):
    """
    Compute balanced class weights from the training split.
    Returns dict {class_idx: weight} for use in model.fit().
    """
    df = pd.read_csv("data/splits/train.csv")
    classes      = config["data"]["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(label_to_idx).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=y
    )
    weight_dict = dict(enumerate(weights))

    print("\nClass weights:")
    for cls, idx in label_to_idx.items():
        print(f"  {cls:30s} → {weight_dict[idx]:.4f}")
    return weight_dict


def get_callbacks(model_name, config):
    save_dir = Path("results/saved_models") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "best.weights.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,        # ← fix: weights only avoids serialization error
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["training"]["lr_reduce_factor"],
            patience=config["training"]["lr_reduce_patience"],
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(save_dir / "training_log.csv")
        ),
    ]
def train(model_name, config):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*55}\n")

    classes     = config["data"]["classes"]
    num_classes = len(classes)
    lr          = config["training"]["learning_rate"]
    epochs      = config["training"]["epochs"]
    warmup      = config["models"]["pretrained"]["warmup_epochs"]
    fine_lr     = config["models"]["pretrained"]["fine_tune_lr"]

    train_ds, _ = make_dataset(
        "data/splits/train.csv", config, augment=True, shuffle=True)
    val_ds, _   = make_dataset(
        "data/splits/val.csv",   config, augment=False)

    weights = compute_weights(config)

    if model_name == "custom_cnn":
        model = build_custom_cnn(num_classes, config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=weights,
            callbacks=get_callbacks(model_name, config)
        )

    else:
        model, base = build_pretrained(model_name, num_classes, config)

        # Stage 1 — head only
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(f"\n--- Stage 1: training head only ({warmup} epochs) ---")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup,
            class_weight=weights
        )

        # Stage 2 — fine-tune
        model = unfreeze_for_finetuning(model, base, config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(fine_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(f"\n--- Stage 2: fine-tuning (up to {epochs} epochs) ---")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=weights,
            callbacks=get_callbacks(model_name, config)
        )

    print(f"\nDone. Best model saved to "
          f"results/saved_models/{model_name}/best.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["custom_cnn", "vgg16", "vgg19", "resnet50", "efficientnetb0"]
    )
    args = parser.parse_args()

    cfg = load_config()
    build_split_csv(cfg)
    train(args.model, cfg)