# src/train.py

import os
import tensorflow as tf

from src.config import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DATASET_DIR,
    IMAGES_DIR,
    MASKS_DIR,
    MODEL_PATH,
    WEIGHTS_DIR,
)

from src.dataset import get_dataset
from src.model import build_model


def main():
    # -----------------------------
    # Prepare paths
    # -----------------------------
    images_path = os.path.join(DATASET_DIR, IMAGES_DIR)
    masks_path = os.path.join(DATASET_DIR, MASKS_DIR)

    print("Images path:", images_path)
    print("Masks path:", masks_path)

    # -----------------------------
    # Create dataset
    # -----------------------------
    train_dataset = get_dataset(images_path, masks_path, shuffle=True)

    # -----------------------------
    # Build model
    # -----------------------------
    model = build_model()
    model.summary()

    # -----------------------------
    # Compile model
    # -----------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Binary segmentation -> Binary Crossentropy
    loss = tf.keras.losses.BinaryCrossentropy()

    # Metrics: accuracy + IoU
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.MeanIoU(num_classes=2, name="iou"),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    # -----------------------------
    # Callbacks (save best model)
    # -----------------------------
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb],
    )

    print("Training finished.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()