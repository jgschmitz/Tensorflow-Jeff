import numpy as np
from PIL import Image as PILImage
from IPython.display import display

def display_fractal(ns, iters):
    # Vectorized color mapping
    phase = (2 * np.pi * ns / 20.0)[..., None]
    img = np.stack([
        10 + 20 * np.cos(phase),
        30 + 50 * np.sin(phase),
        155 - 80 * np.cos(phase)
    ], axis=2).squeeze()

    img[ns == iters] = 0 # Darken the center
    img = np.uint8(np.clip(img, 0, 255))
    display(PILImage.fromarray(img))

# 1. Faster Grid Generation
y, x = np.ogrid[-1.3:1.3:0.005, -2:1:0.005]
c = x + 1j * y
z = np.zeros_like(c)
ns = np.zeros(c.shape, dtype=int)

# 2. The Tight Loop (NumPy Vectorization)
iters = 200
for i in range(iters):
    z = z**2 + c
    mask = np.abs(z) < 4.0 # Escape radius
    ns += mask
    z[~mask] = 0 # Prevent overflows in diverged points

display_fractal(ns, iters)
