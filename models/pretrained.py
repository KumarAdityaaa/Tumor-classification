import tensorflow as tf


BACKBONES = {
    "vgg16":          tf.keras.applications.VGG16,
    "vgg19":          tf.keras.applications.VGG19,
    "resnet50":       tf.keras.applications.ResNet50,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
}

PREPROCESSORS = {
    "vgg16":          tf.keras.applications.vgg16.preprocess_input,
    "vgg19":          tf.keras.applications.vgg19.preprocess_input,
    "resnet50":       tf.keras.applications.resnet50.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
}


def build_pretrained(name, num_classes, config):
    """
    Stage 1: freeze base, train classification head only.
    Stage 2: unfreeze last N layers, fine-tune with low LR.
    Returns (model, base) frozen for stage-1 training.
    """
    img_size = config["data"]["img_size"]
    name = name.lower()

    base = BACKBONES[name](
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = PREPROCESSORS[name](inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    return model, base


def unfreeze_for_finetuning(model, base, config):
    """Unfreeze last N layers of the base for stage-2 fine-tuning."""
    n = config["models"]["pretrained"]["fine_tune_layers"]
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    total_trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Unfroze last {n} layers of {base.name} "
          f"({total_trainable} trainable layers total)")
    return model