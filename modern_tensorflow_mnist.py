"""
Modern TensorFlow / Keras MNIST example.

Run:
    python modern_tensorflow_mnist.py --epochs 10

Demonstrates:
- tf.keras CNN model
- efficient tf.data pipeline
- reproducible train/validation split
- learning-rate reduction
- early stopping
- best-model checkpointing
- test-set evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


DEFAULT_SEED = 42
VALIDATION_SIZE = 10_000
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28, 1)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a TensorFlow MNIST classifier."
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=10,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=128,
        help="Training and evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=positive_int,
        default=3,
        help="Epochs without validation improvement before stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/mnist_classifier"),
        help="Directory used to store the best model.",
    )
    return parser.parse_args()


def normalize_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    image = tf.ensure_shape(image, IMAGE_SHAPE)

    label = tf.cast(label, tf.int32)
    return image, label


def build_dataset(
    images: tf.Tensor,
    labels: tf.Tensor,
    batch_size: int,
    *,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.map(
        normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # MNIST fits comfortably in memory, so caching avoids repeating
    # normalization work during every epoch.
    dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=IMAGE_SHAPE, name="image")

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    # Fewer parameters than Flatten followed by a large Dense layer.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="digit",
    )(x)

    return tf.keras.Model(inputs, outputs, name="mnist_cnn")


def main() -> None:
    args = parse_args()

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")

    args.model_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)

    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = tf.keras.datasets.mnist.load_data()

    validation_images = train_images[-VALIDATION_SIZE:]
    validation_labels = train_labels[-VALIDATION_SIZE:]

    train_images = train_images[:-VALIDATION_SIZE]
    train_labels = train_labels[:-VALIDATION_SIZE]

    train_ds = build_dataset(
        train_images,
        train_labels,
        args.batch_size,
        training=True,
        seed=args.seed,
    )
    validation_ds = build_dataset(
        validation_images,
        validation_labels,
        args.batch_size,
        training=False,
        seed=args.seed,
    )
    test_ds = build_dataset(
        test_images,
        test_labels,
        args.batch_size,
        training=False,
        seed=args.seed,
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    model.summary()

    checkpoint_path = args.model_dir / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate the saved artifact—not merely the current in-memory model.
    best_model = tf.keras.models.load_model(checkpoint_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)

    print("\nFinal test results")
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2%}")
    print(f"Model:    {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
