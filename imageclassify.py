import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.utils import load_img, img_to_array

NUM_TOP_PREDICTIONS = 5


def load_and_preprocess_image(img_path):
    """Load and preprocess an image for InceptionV3."""
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def run_inference_on_image(model, img_path):
    """Run inference and print top predictions."""
    img = load_and_preprocess_image(img_path)
    preds = model.predict(img, verbose=0)

    print("\nPredicted:")
    predictions = decode_predictions(preds, top=NUM_TOP_PREDICTIONS)[0]
    for _, label, prob in predictions:
        print(f"{label}: {prob * 100:.2f}%")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")

    print("Loading InceptionV3 model...")
    model = InceptionV3(weights="imagenet")

    run_inference_on_image(model, image_path)


if __name__ == "__main__":
    main()
