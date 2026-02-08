# Core libraries
import tensorflow as tf
import numpy as np

# Visualization
from cStringIO import StringIO
from IPython.display import Image, display
import PIL.Image


def display_fractal(a, fmt='jpeg'):
    """Render iteration counts as a colorful fractal image."""
    phase = (6.28 * a / 20.0)[..., None]
    img = np.concatenate([
        10 + 20 * np.cos(phase),
        30 + 50 * np.sin(phase),
        155 - 80 * np.cos(phase)
    ], axis=2)

    img[a == a.max()] = 0
    img = np.uint8(np.clip(img, 0, 255))

    buf = StringIO()
    PIL.Image.fromarray(img).save(buf, fmt)
    display(Image(data=buf.getvalue()))


# Interactive session
sess = tf.InteractiveSession()

# Complex grid
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j * Y

# TensorFlow setup
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

tf.initialize_all_variables().run()

# Mandelbrot iteration
zs_next = zs * zs + xs
not_diverged = tf.complex_abs(zs_next) < 4

step = tf.group(
    zs.assign(zs_next),
    ns.assign_add(tf.cast(not_diverged, tf.float32))
)

# Run iterations
for _ in range(200):
    step.run()

# Display result
display_fractal(ns.eval())
