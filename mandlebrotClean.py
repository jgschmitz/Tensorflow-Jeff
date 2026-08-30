import numpy as np
from PIL import Image as PILImage
from IPython.display import display


def display_fractal(ns, interior):
    phase = 2 * np.pi * ns / 20.0

    img = np.stack([
        10 + 20 * np.cos(phase),
        30 + 50 * np.sin(phase),
        155 - 80 * np.cos(phase)
    ], axis=-1)

    # Mandelbrot set itself
    img[interior] = 0

    img = np.uint8(np.clip(img, 0, 255))
    display(PILImage.fromarray(img))


# Grid
y, x = np.ogrid[-1.3:1.3:0.005, -2:1:0.005]
c = x + 1j * y

z = np.zeros_like(c)
ns = np.zeros(c.shape, dtype=np.int32)
active = np.ones(c.shape, dtype=bool)

iters = 200

for i in range(iters):
    z[active] = z[active] ** 2 + c[active]

    escaped = active & (np.abs(z) > 2.0)

    # Record exactly when each point escaped
    ns[escaped] = i + 1

    active[escaped] = False

# Anything still active after all iterations is in/near the set
ns[active] = iters

display_fractal(ns, active)
