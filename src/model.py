"""CNN classifier definition (matches Table 6.2 of the report)."""
from tensorflow.keras import layers, models
import config

FRAMES = config.N_SAMPLES // config.HOP_LENGTH + 1


def build_model(n_classes=len(config.CLASSES)):
    m = models.Sequential([
        layers.Input(shape=(config.N_MELS, FRAMES, 1)),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
