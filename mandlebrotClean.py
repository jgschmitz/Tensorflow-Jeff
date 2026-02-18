import numpy as np
import tensorflow as tf
from io import BytesIO
from IPython.display import Image, display
from PIL import Image as PILImage


def display_fractal(a, fmt="png"):
    phase = (2 * np.pi * a / 20.0)[..., None]
    img = np.concatenate(
        [
            10 + 20 * np.cos(phase),
            30 + 50 * np.sin(phase),
            155 - 80 * np.cos(phase),
        ],
        axis=2,
    )

    img[a == a.max()] = 0
    img = np.uint8(np.clip(img, 0, 255))

    buf = BytesIO()
    PILImage.fromarray(img).save(buf, format=fmt.upper())
    display(Image(data=buf.getvalue()))


# Complex grid
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
xs_np = (X + 1j * Y).astype(np.complex64)

xs = tf.constant(xs_np)

def mandelbrot(xs, iters=200, escape_radius=4.0):
    zs0 = tf.zeros_like(xs)
    ns0 = tf.zeros(xs.shape, dtype=tf.int32)

    def cond(i, zs, ns):
        return i < iters

    def body(i, zs, ns):
        zs_next = zs * zs + xs
        not_diverged = tf.abs(zs_next) < escape_radius
        ns_next = ns + tf.cast(not_diverged, tf.int32)
        # Optional: freeze diverged points to avoid overflow growth
        zs_next = tf.where(not_diverged, zs_next, zs)
        return i + 1, zs_next, ns_next

    _, _, ns = tf.while_loop(cond, body, [0, zs0, ns0], parallel_iterations=1)
    return ns

ns = mandelbrot(xs, iters=200)

with tf.Session() as sess:
    ns_out = sess.run(ns)

display_fractal(ns_out)
