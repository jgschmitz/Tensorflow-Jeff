#updated to 4.0 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def plot_hyperplane(classifier, min_x, max_x, linestyle, label):
    """Plot a linear decision boundary for a trained classifier."""
    weights = classifier.coef_[0]
    slope = -weights[0] / weights[1]

    x_values = np.linspace(min_x - 5, max_x + 5)
    y_values = slope * x_values - classifier.intercept_[0] / weights[1]

    plt.plot(x_values, y_values, linestyle, label=label)


def transform_data(X, Y, method):
    """Transform X using either PCA or CCA."""
    if method == "pca":
        return PCA(n_components=2).fit_transform(X)

    if method == "cca":
        return CCA(n_components=2).fit(X, Y).transform(X)

    raise ValueError(f"Unknown transformation method: {method}")


def plot_subfigure(X, Y, subplot, title, transform):
    """Transform the data, train classifiers, and create one subplot."""
    X_transformed = transform_data(X, Y, transform)

    min_x, max_x = X_transformed[:, 0].min(), X_transformed[:, 0].max()
    min_y, max_y = X_transformed[:, 1].min(), X_transformed[:, 1].max()

    # Train one classifier for each class.
    classifier = OneVsRestClassifier(SVC(kernel="linear"))
    classifier.fit(X_transformed, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    # Plot all samples.
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        s=40,
        c="gray",
        edgecolors="black",
    )

    # Highlight samples belonging to each class.
    class_1 = Y[:, 0].astype(bool)
    class_2 = Y[:, 1].astype(bool)

    plt.scatter(
        X_transformed[class_1, 0],
        X_transformed[class_1, 1],
        s=160,
        edgecolors="blue",
        facecolors="none",
        linewidths=2,
        label="Class 1",
    )

    plt.scatter(
        X_transformed[class_2, 0],
        X_transformed[class_2, 1],
        s=80,
        edgecolors="orange",
        facecolors="none",
        linewidths=2,
        label="Class 2",
    )

    # Plot decision boundaries.
    plot_hyperplane(
        classifier.estimators_[0],
        min_x,
        max_x,
        "k--",
        "Boundary\nfor class 1",
    )

    plot_hyperplane(
        classifier.estimators_[1],
        min_x,
        max_x,
        "k-.",
        "Boundary\nfor class 2",
    )

    plt.xticks([])
    plt.yticks([])

    # Add some padding around the data.
    x_padding = 0.5 * abs(max_x)
    y_padding = 0.5 * abs(max_y)

    plt.xlim(min_x - x_padding, max_x + x_padding)
    plt.ylim(min_y - y_padding, max_y + y_padding)

    if subplot == 2:
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.legend(loc="upper left")


# ------------------------------------------------------------
# Generate data with unlabeled samples
# ------------------------------------------------------------

X, Y = make_multilabel_classification(
    n_classes=2,
    n_labels=1,
    allow_unlabeled=True,
    random_state=1,
)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")


# ------------------------------------------------------------
# Generate data without unlabeled samples
# ------------------------------------------------------------

X, Y = make_multilabel_classification(
    n_classes=2,
    n_labels=1,
    allow_unlabeled=False,
    random_state=1,
)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")


plt.subplots_adjust(
    left=0.04,
    bottom=0.02,
    right=0.97,
    top=0.94,
    wspace=0.09,
    hspace=0.20,
)

plt.show()
