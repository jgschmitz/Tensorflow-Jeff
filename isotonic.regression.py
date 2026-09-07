import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

# Data generation
x = np.arange(100)
y = np.random.default_rng(0).integers(-50, 50, 100) + 50 * np.log1p(x)

# Model fitting
y_iso = IsotonicRegression().fit_transform(x, y)
y_lin = LinearRegression().fit(x[:, None], y).predict(x[:, None])

# Plotting
plt.figure(figsize=(10, 6))
plt.vlines(x, y, y_iso, colors='gray', linewidths=0.5, zorder=0)
plt.scatter(x, y, c='r', s=20, label='Data')
plt.plot(x, y_iso, 'b.-', label='Isotonic Fit', ms=8)
plt.plot(x, y_lin, 'g-', label='Linear Fit')

plt.title('Isotonic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
