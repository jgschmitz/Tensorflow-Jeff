"""
Modern TensorFlow 2 / Keras MNIST example.

Run:
    python modern_tensorflow_mnist.py --epochs 5

This script demonstrates a compact, up-to-date TensorFlow workflow:
- tf.keras model definition
- tf.data input pipeline
- validation split
- model checkpointing
- test-set evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modern TensorFlow MNIST classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/mnist_classifier"),
        help="Directory for saved model checkpoints.",
    )
    return parser.parse_args()


def normalize_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    return image, label


def build_dataset(
    images: tf.Tensor,
    labels: tf.Tensor,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=10_000)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(42)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    validation_size = 10_000
    validation_images = train_images[-validation_size:]
    validation_labels = train_labels[-validation_size:]
    train_images = train_images[:-validation_size]
    train_labels = train_labels[:-validation_size]

    train_ds = build_dataset(train_images, train_labels, args.batch_size, training=True)
    validation_ds = build_dataset(validation_images, validation_labels, args.batch_size, training=False)
    test_ds = build_dataset(test_images, test_labels, args.batch_size, training=False)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    checkpoint_path = args.model_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
