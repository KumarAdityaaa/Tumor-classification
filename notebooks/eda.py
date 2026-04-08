"""
Exploratory Data Analysis — Brain Tumor MRI Dataset
Generates all figures needed for the Dataset section of the paper.
Run: python notebooks/eda.py
Output: results/figures/eda/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import load_config

OUT_DIR = Path("results/figures/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

CLASS_COLORS = {
    "glioma_tumor":     "#1D9E75",
    "meningioma_tumor": "#378ADD",
    "no_tumor":         "#EF9F27",
    "pituitary_tumor":  "#D85A30",
}

CLASS_LABELS = {
    "glioma_tumor":     "Glioma",
    "meningioma_tumor": "Meningioma",
    "no_tumor":         "No Tumor",
    "pituitary_tumor":  "Pituitary",
}


def load_split_info(config):
    splits = {}
    for split in ["train", "val", "test"]:
        path = Path(config["data"]["splits_dir"]) / f"{split}.csv"
        splits[split] = pd.read_csv(path)
    return splits


# ── EDA Fig 1: Class distribution across splits ──────────────────────────────
def fig_class_distribution(splits, classes):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    split_names = ["train", "val", "test"]
    titles      = ["Training set", "Validation set", "Test set"]

    for ax, split, title in zip(axes, split_names, titles):
        df     = splits[split]
        counts = df["label"].value_counts().reindex(classes).fillna(0)
        labels = [CLASS_LABELS[c] for c in classes]
        colors = [CLASS_COLORS[c] for c in classes]
        bars   = ax.bar(labels, counts.values, color=colors, alpha=0.88, width=0.6)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2, int(val),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} (n={len(df)})")
        ax.set_ylabel("Image count")
        ax.tick_params(axis="x", rotation=15)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("EDA Figure 1 — Class distribution across train / val / test splits",
                 y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "eda_fig1_class_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── EDA Fig 2: Sample MRI grid (4 classes × 5 samples) ──────────────────────
def fig_sample_grid(splits, classes, config, n_samples=5):
    raw_dir = Path(config["data"]["raw_dir"])
    fig, axes = plt.subplots(len(classes), n_samples,
                             figsize=(n_samples * 2.2, len(classes) * 2.2))
    random.seed(42)

    for row, cls in enumerate(classes):
        df_cls = splits["train"][splits["train"]["label"] == cls]
        sample_paths = random.sample(list(df_cls["path"].values),
                                     min(n_samples, len(df_cls)))
        for col, img_path in enumerate(sample_paths):
            ax = axes[row][col]
            try:
                img = Image.open(img_path).convert("L")
                ax.imshow(img, cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_LABELS[cls], rotation=90,
                              fontsize=10, fontweight="bold",
                              labelpad=10)
                ax.yaxis.set_label_coords(-0.15, 0.5)
        axes[row][0].set_visible(True)

    fig.suptitle("EDA Figure 2 — Sample MRI images per class",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / "eda_fig2_sample_grid.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── EDA Fig 3: Pixel intensity distributions per class ──────────────────────
def fig_pixel_intensity(splits, classes, n_per_class=80):
    fig, ax = plt.subplots(figsize=(8, 4))
    random.seed(42)

    for cls in classes:
        df_cls   = splits["train"][splits["train"]["label"] == cls]
        sample   = random.sample(list(df_cls["path"].values),
                                 min(n_per_class, len(df_cls)))
        all_vals = []
        for p in sample:
            try:
                img = np.array(Image.open(p).convert("L").resize((128, 128)))
                all_vals.append(img.flatten())
            except Exception:
                continue
        if not all_vals:
            continue
        vals = np.concatenate(all_vals)
        ax.hist(vals, bins=64, alpha=0.45, density=True,
                color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])

    ax.set_xlabel("Pixel intensity (0–255)")
    ax.set_ylabel("Density")
    ax.set_title("EDA Figure 3 — Pixel intensity distribution per class")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_DIR / "eda_fig3_pixel_intensity.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── EDA Fig 4: Image size distribution ──────────────────────────────────────
def fig_image_sizes(splits, classes, n_per_class=100):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    random.seed(42)
    widths, heights = [], []

    for cls in classes:
        df_cls = splits["train"][splits["train"]["label"] == cls]
        sample = random.sample(list(df_cls["path"].values),
                               min(n_per_class, len(df_cls)))
        for p in sample:
            try:
                img = Image.open(p)
                widths.append(img.width)
                heights.append(img.height)
            except Exception:
                continue

    axes[0].hist(widths,  bins=30, color="#378ADD", alpha=0.8)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image width distribution")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(heights, bins=30, color="#1D9E75", alpha=0.8)
    axes[1].set_xlabel("Height (px)")
    axes[1].set_title("Image height distribution")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("EDA Figure 4 — Raw image size distribution (before resizing)",
                 y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "eda_fig4_image_sizes.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── EDA Table: Split summary ─────────────────────────────────────────────────
def save_split_table(splits, classes):
    rows = []
    for cls in classes:
        row = {"Class": CLASS_LABELS[cls]}
        total = 0
        for split in ["train", "val", "test"]:
            count = (splits[split]["label"] == cls).sum()
            row[split.capitalize()] = count
            total += count
        row["Total"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    totals_row = {"Class": "Total"}
    for col in ["Train", "Val", "Test", "Total"]:
        totals_row[col] = df[col].sum()
    df = pd.concat([df, pd.DataFrame([totals_row])], ignore_index=True)

    path = OUT_DIR / "eda_table_split_summary.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    print("\n" + df.to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg     = load_config()
    classes = cfg["data"]["classes"]
    splits  = load_split_info(cfg)

    print("Generating EDA figures...\n")
    fig_class_distribution(splits, classes)
    fig_sample_grid(splits, classes, cfg)
    fig_pixel_intensity(splits, classes)
    fig_image_sizes(splits, classes)
    save_split_table(splits, classes)
    print(f"\nAll EDA outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()