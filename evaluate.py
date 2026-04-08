import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.dataset import load_config, make_dataset
from models.custom_cnn import build_custom_cnn
from models.pretrained import build_pretrained, unfreeze_for_finetuning


def evaluate(model_name, config):
    classes     = config["data"]["classes"]
    num_classes = len(classes)

    keras_path   = Path(f"results/saved_models/{model_name}/best.keras")
    weights_path = Path(f"results/saved_models/{model_name}/best.weights.h5")

    if keras_path.exists():
        model = tf.keras.models.load_model(str(keras_path))
        print(f"Loaded: {keras_path}\n")
    elif weights_path.exists():
        if model_name == "custom_cnn":
            model = build_custom_cnn(num_classes, config)
        else:
            model, base = build_pretrained(model_name, num_classes, config)
            model = unfreeze_for_finetuning(model, base, config)
        model.load_weights(str(weights_path))
        print(f"Loaded weights: {weights_path}\n")
    else:
        raise FileNotFoundError(
            f"No saved model found for '{model_name}'. Train it first.")

    test_ds, _ = make_dataset(
        "data/splits/test.csv", config, augment=False)

    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred_list.append(probs)
        y_true_list.append(y_batch.numpy())

    y_true      = np.concatenate(y_true_list)
    y_pred_prob = np.concatenate(y_pred_list)
    y_true_idx  = np.argmax(y_true, axis=1)
    y_pred_idx  = np.argmax(y_pred_prob, axis=1)

    out_dir = Path(f"results/figures/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Classification report ──────────────────────────────────────
    report = classification_report(
        y_true_idx, y_pred_idx,
        target_names=classes, digits=4
    )
    print(f"Classification Report — {model_name}\n")
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # ── Confusion matrix ───────────────────────────────────────────
    cm      = confusion_matrix(y_true_idx, y_pred_idx)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax
        )
        ax.set_title(f"Confusion matrix ({title}) — {model_name}")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/confusion_matrix.png")

    # ── ROC curves (one-vs-rest) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/roc_curves.png")

    # ── Training curves ────────────────────────────────────────────
    log_path = Path(f"results/saved_models/{model_name}/training_log.csv")
    if log_path.exists():
        log = pd.read_csv(log_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes, ["accuracy", "loss"]):
            ax.plot(log[metric],          label="Train")
            ax.plot(log[f"val_{metric}"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} — {model_name}")
            ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out_dir}/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["custom_cnn", "vgg16", "vgg19", "resnet50", "efficientnetb0"]
    )
    args = parser.parse_args()

    cfg = load_config()
    evaluate(args.model, cfg)