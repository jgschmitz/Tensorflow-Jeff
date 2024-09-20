import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Constants
MODEL_DIR = '/tmp/imagenet'
IMAGE_FILE = 'your_image_path.jpg'  # Update this with the path to your image
NUM_TOP_PREDICTIONS = 5

# Load the pre-trained InceptionV3 model + weights trained on ImageNet
model = InceptionV3(weights='imagenet')

def load_and_preprocess_image(img_path):
    """Loads and preprocesses an image for InceptionV3."""
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def run_inference_on_image(img_path):
    """Runs inference on an image and prints top predictions."""
    img = load_and_preprocess_image(img_path)
    preds = model.predict(img)

    # Decode the results into a list of tuples (class, description, probability)
    # (e.g. [('n01440764', 'tench', 0.8927636), ...])
    print('Predicted:')
    predictions = decode_predictions(preds, top=NUM_TOP_PREDICTIONS)[0]
    for pred in predictions:
        print(f"{pred[1]}: {pred[2] * 100:.2f}%")

if __name__ == '__main__':
    # Check if the image file exists
    if not os.path.exists(IMAGE_FILE):
        raise FileNotFoundError(f"Image file {IMAGE_FILE} does not exist.")
    
    # Run inference
    run_inference_on_image(IMAGE_FILE)
