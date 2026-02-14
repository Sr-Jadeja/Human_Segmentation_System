# src/inference.py

import os
import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, MODEL_PATH

def preprocess_image(image_path):
    # Load image, resize, normalize, and add batch dimension
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    # Add batch dimension: (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess_mask(mask):
    # Convert model output to 0-255 mask image
    mask = mask[0]  # remove batch dimension -> (H, W, 1)
    mask = mask.squeeze()  # (H, W)

    # Threshold to binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    input_image_path = "assets/test.jpg"
    output_mask_path = "assets/pred_mask.png"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    print("Loading full model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Loading image:", input_image_path)
    img = preprocess_image(input_image_path)

    print("Running inference...")
    pred = model.predict(img)

    mask = postprocess_mask(pred)

    print("Saving predicted mask to:", output_mask_path)
    cv2.imwrite(output_mask_path, mask)

    print("Done! Predicted mask saved as:", output_mask_path)


if __name__ == "__main__":
    main()