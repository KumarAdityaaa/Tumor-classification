"""
Publication-ready figures for journal paper.
Run after all 5 models are evaluated:
    python scripts/generate_figures.py
Outputs → results/figures/paper/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import load_config, make_dataset
from models.custom_cnn import build_custom_cnn
from models.pretrained import build_pretrained, unfreeze_for_finetuning

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["vgg16", "vgg19", "resnet50", "efficientnetb0", "custom_cnn"]
MODEL_LABELS = {
    "vgg16":          "VGG16",
    "vgg19":          "VGG19",
    "resnet50":       "ResNet50",
    "efficientnetb0": "EfficientNetB0",
    "custom_cnn":     "Custom CNN",
}

PALETTE = {
    "vgg16":          "#1D9E75",
    "vgg19":          "#378ADD",
    "resnet50":       "#EF9F27",
    "efficientnetb0": "#D85A30",
    "custom_cnn":     "#7F77DD",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model_for_eval(model_name, config):
    num_classes = len(config["data"]["classes"])
    keras_path   = Path(f"results/saved_models/{model_name}/best.keras")
    weights_path = Path(f"results/saved_models/{model_name}/best.weights.h5")

    if keras_path.exists():
        return tf.keras.models.load_model(str(keras_path))
    elif weights_path.exists():
        if model_name == "custom_cnn":
            model = build_custom_cnn(num_classes, config)
        else:
            model, base = build_pretrained(model_name, num_classes, config)
            model = unfreeze_for_finetuning(model, base, config)
        model.load_weights(str(weights_path))
        return model
    else:
        raise FileNotFoundError(f"No saved model for {model_name}")


def get_predictions(model, test_ds):
    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred_list.append(probs)
        y_true_list.append(y_batch.numpy())
    y_true      = np.concatenate(y_true_list)
    y_pred_prob = np.concatenate(y_pred_list)
    return y_true, y_pred_prob


def extract_metrics(y_true, y_pred_prob, classes):
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)
    report = classification_report(
        y_true_idx, y_pred_idx,
        target_names=classes, output_dict=True
    )
    return {
        "accuracy":  report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall":    report["macro avg"]["recall"],
        "f1":        report["macro avg"]["f1-score"],
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
        "cm":        confusion_matrix(y_true_idx, y_pred_idx),
        "y_true":    y_true,
        "y_pred_prob": y_pred_prob,
    }


# ── Figure 1: Overall metrics bar chart ─────────────────────────────────────
def fig_overall_metrics(results, classes):
    metrics   = ["accuracy", "precision", "recall", "f1"]
    labels    = ["Accuracy", "Precision", "Recall", "F1 Score"]
    n_models  = len(MODELS)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    width     = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(MODELS):
        vals = [results[m][met] for met in metrics]
        bars = ax.bar(
            x + i * width, vals, width,
            label=MODEL_LABELS[m],
            color=PALETTE[m], alpha=0.88
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7, rotation=90
            )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Figure 1 — Overall performance comparison (test set, n=474)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_DIR / "fig1_overall_metrics.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 2: Per-class F1 heatmap ──────────────────────────────────────────
def fig_perclass_f1(results, classes):
    short = [c.replace("_tumor", "").replace("no_tumor", "No tumor").title()
             for c in classes]
    data = np.array([
        [results[m]["per_class_f1"][c] for c in classes]
        for m in MODELS
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="YlGn", vmin=0.65, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1 Score")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=20, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])

    for i in range(len(MODELS)):
        for j in range(len(classes)):
            val = data[i, j]
            color = "white" if val < 0.80 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="500")

    ax.set_title("Figure 2 — Per-class F1 score heatmap")
    plt.tight_layout()
    path = OUT_DIR / "fig2_perclass_f1_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 3: Confusion matrices (best model = VGG16) ───────────────────────
def fig_confusion_matrices(results, classes):
    short = [c.replace("_tumor", "").replace("no_tumor", "No tumor").title()
             for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm      = results["vgg16"]["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["(a) Raw counts", "(b) Row-normalized"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=short, yticklabels=short,
            ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"size": 9}
        )
        ax.set_title(f"Figure 3 — VGG16 confusion matrix {title}")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = OUT_DIR / "fig3_confusion_matrix_vgg16.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 4: ROC curves — all models on one plot per class ─────────────────
def fig_roc_curves(results, classes):
    short = []
    for c in classes:
        if c == "no_tumor":
            short.append("No Tumor")
        else:
            short.append(c.replace("_tumor", "").title())
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)

    for j, (cls, label) in enumerate(zip(classes, short)):
        ax = axes[j]
        for m in MODELS:
            y_true      = results[m]["y_true"]
            y_pred_prob = results[m]["y_pred_prob"]
            fpr, tpr, _ = roc_curve(y_true[:, j], y_pred_prob[:, j])
            auc = roc_auc_score(y_true[:, j], y_pred_prob[:, j])
            ax.plot(fpr, tpr, label=f"{MODEL_LABELS[m]} ({auc:.3f})",
                    color=PALETTE[m], lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("FPR")
        if j == 0:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Figure 4 — ROC curves per class (all models)", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "fig4_roc_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 5: Training curves — all models ──────────────────────────────────
def fig_training_curves():
    fig, axes = plt.subplots(2, len(MODELS), figsize=(4 * len(MODELS), 7))

    for col, m in enumerate(MODELS):
        log_path = Path(f"results/saved_models/{m}/training_log.csv")
        if not log_path.exists():
            print(f"  No training log for {m}, skipping.")
            continue
        log = pd.read_csv(log_path)
        color = PALETTE[m]

        for row, metric in enumerate(["accuracy", "loss"]):
            ax = axes[row][col]
            ax.plot(log[metric],          label="Train",
                    color=color, lw=1.5)
            ax.plot(log[f"val_{metric}"], label="Val",
                    color=color, lw=1.5, linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize() if col == 0 else "")
            ax.set_title(MODEL_LABELS[m] if row == 0 else "")
            ax.legend(fontsize=7)
            ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Figure 5 — Training and validation curves", y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig5_training_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Table 1: Full metrics CSV (paste into paper) ────────────────────────────
def save_metrics_table(results, classes):
    rows = []
    for m in MODELS:
        r = results[m]
        row = {
            "Model":     MODEL_LABELS[m],
            "Accuracy":  f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1 (macro)":f"{r['f1']:.4f}",
        }
        for c in classes:
            short = c.replace("_tumor", "").replace("no_", "No ").title()
            row[f"F1 ({short})"] = f"{r['per_class_f1'][c]:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    path = OUT_DIR / "table1_full_metrics.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    print("\n" + df.to_string(index=False))
    return df


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    cfg     = load_config()
    classes = cfg["data"]["classes"]

    test_ds, _ = make_dataset(
        "data/splits/test.csv", cfg, augment=False)

    results = {}
    for m in MODELS:
        print(f"\nLoading {m}...")
        model = load_model_for_eval(m, cfg)
        y_true, y_pred_prob = get_predictions(model, test_ds)
        results[m] = extract_metrics(y_true, y_pred_prob, classes)
        print(f"  Accuracy: {results[m]['accuracy']:.4f} | "
              f"Macro F1: {results[m]['f1']:.4f}")
        tf.keras.backend.clear_session()

    print("\nGenerating figures...")
    fig_overall_metrics(results, classes)
    fig_perclass_f1(results, classes)
    fig_confusion_matrices(results, classes)
    fig_roc_curves(results, classes)
    fig_training_curves()
    save_metrics_table(results, classes)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()