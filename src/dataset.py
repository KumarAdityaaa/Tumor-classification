import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_split_csv(config):
    """
    Walk data/raw/<classname>/, collect all image paths + labels,
    then produce stratified train/val/test CSVs in data/splits/.
    Skips if CSVs already exist.
    """
    splits_dir = Path(config["data"]["splits_dir"])
    train_csv = splits_dir / "train.csv"
    if train_csv.exists():
        print("Split CSVs already exist — skipping.")
        return

    raw_dir = Path(config["data"]["raw_dir"])
    classes = config["data"]["classes"]
    seed = config["data"]["random_seed"]

    records = []
    for cls in classes:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append({"path": str(img_path), "label": cls})

    df = pd.DataFrame(records)
    print(f"\nTotal images found: {len(df)}")
    print(df["label"].value_counts().to_string())

    val_size  = config["data"]["val_split"]
    test_size = config["data"]["test_split"]

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["label"], random_state=seed
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv",   index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"\nSplit saved:")
    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")


def make_dataset(csv_path, config, augment=False, shuffle=False):
    """
    Build a tf.data.Dataset from a CSV of (path, label) rows.
    """
    df = pd.read_csv(csv_path)
    classes      = config["data"]["classes"]
    img_size     = config["data"]["img_size"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    aug_cfg      = config["augmentation"]
    num_classes  = len(classes)

    paths  = df["path"].values
    labels = df["label"].map(label_to_idx).values

    # Create augmentation layers ONCE outside the map function
    rotation_layer = tf.keras.layers.RandomRotation(
        aug_cfg["rotation_range"] / 360)
    zoom_layer = tf.keras.layers.RandomZoom(
        aug_cfg["zoom_range"])

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return img, label

    def augment_fn(img, label):
        if aug_cfg["horizontal_flip"]:
            img = tf.image.random_flip_left_right(img)
        bmin, bmax = aug_cfg["brightness_range"]
        img = tf.image.random_brightness(img, max_delta=(bmax - bmin) / 2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        img = tf.expand_dims(img, 0)
        img = rotation_layer(img, training=True)
        img = zoom_layer(img, training=True)
        img = tf.squeeze(img, 0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config["training"]["batch_size"])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(df)