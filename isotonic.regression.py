"""
Isotonic Regression Example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

# Parameters
n = 100
x = np.arange(n)
rs = check_random_state(0)

# Generate random data
y = rs.randint(-50, 50, size=n) + 50 * np.log1p(x)

# Fit models
isotonic_model = IsotonicRegression()
y_isotonic = isotonic_model.fit_transform(x, y)

linear_model = LinearRegression()
linear_model.fit(x[:, np.newaxis], y)

# Create line segments for visualization
segments = np.stack([np.column_stack([x, y]), np.column_stack([x, y_isotonic])], axis=1)
lc = LineCollection(segments, zorder=0, linewidths=0.5, colors='gray')

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data', s=20)
plt.plot(x, y_isotonic, 'b.-', label='Isotonic Fit', markersize=8)
plt.plot(x, linear_model.predict(x[:, np.newaxis]), 'g-', label='Linear Fit')
plt.gca().add_collection(lc)

plt.legend(loc='lower right')
plt.title('Isotonic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid
