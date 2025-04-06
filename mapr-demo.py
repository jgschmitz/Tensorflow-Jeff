from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")  # Adjust for Atlas URI if needed
db = client["tsa_claims"]
train_collection = db["claims_train"]
test_collection = db["claims_test"]

# Load data from MongoDB
def load_data(collection):
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    
    # Example: assume features are columns named "feature1", "feature2", ..., "featureN"
    feature_cols = [col for col in df.columns if col.startswith("feature")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    
    return X, y

X_train, y_train = load_data(train_collection)
X_test, y_test = load_data(test_collection)

# TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(1000)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Use softmax for multi-class
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # use 'sparse_categorical_crossentropy' for multi-class
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save the model
model.save("tsa_claim_model.h5")
