import tensorflow as tf


def build_custom_cnn(num_classes, config):
    """
    Custom CNN with:
    - BatchNormalization after every conv block
    - Global Average Pooling instead of Flatten
    - Configurable dropout and dense units
    """
    dropout_rate = config["models"]["custom_cnn"]["dropout_rate"]
    dense_units  = config["models"]["custom_cnn"]["dense_units"]
    img_size     = config["data"]["img_size"]

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs

    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    return model