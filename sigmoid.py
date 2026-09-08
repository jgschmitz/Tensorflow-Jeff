"""
Probability calibration of a Random Forest classifier
=======================================================
Demonstrates the effect of sigmoid (Platt) calibration on the predicted
class probabilities of a RandomForestClassifier trained on a synthetic
3-class blob dataset, and visualizes the calibration map on the 2-simplex.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
# Prefer FrozenEstimator if available (sklearn >= 1.6); otherwise fall back
# to the deprecated cv="prefit" API for older versions.
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False
RANDOM_STATE = 0
COLORS = ("r", "g", "b")
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_dataset(n_samples=1000, cluster_std=5.0, random_state=RANDOM_STATE):
    """Generate blobs and split into train / valid / train+valid / test."""
    X, y = make_blobs(n_samples=n_samples, random_state=random_state,
                       cluster_std=cluster_std)
    splits = {
        "train": (X[:600], y[:600]),
        "valid": (X[600:800], y[600:800]),
        "train_valid": (X[:800], y[:800]),
        "test": (X[800:], y[800:]),
    }
    return splits
def fit_uncalibrated(X, y, random_state=RANDOM_STATE):
    clf = RandomForestClassifier(n_estimators=25, random_state=random_state,
                                  n_jobs=-1)
    clf.fit(X, y)
    return clf
def fit_calibrated(X_train, y_train, X_valid, y_valid,
                    random_state=RANDOM_STATE):
    """Fit a RF on X_train/y_train, then sigmoid-calibrate on X_valid/y_valid."""
    clf = RandomForestClassifier(n_estimators=25, random_state=random_state,
                                  n_jobs=-1)
    clf.fit(X_train, y_train)
    if HAS_FROZEN_ESTIMATOR:
        sig_clf = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    else:
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_clf.fit(X_valid, y_valid)
    return clf, sig_clf
def safe_normalize(probs):
    """Row-normalize, avoiding division by zero."""
    row_sums = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, row_sums, out=np.zeros_like(probs),
                      where=row_sums != 0)
def draw_simplex_grid(ax, step=0.1):
    """Draw the 2-simplex boundary plus a light triangular grid."""
    ax.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")
    for x in np.arange(0.0, 1.0 + step / 2, step):
        ax.plot([0, x], [x, 0], "k", alpha=0.2)
        ax.plot([0, (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        ax.plot([x, x + (1 - x) / 2], [0, (1 - x) / 2], "k", alpha=0.2)
def annotate_simplex_vertices(ax):
    """Label the vertices, edge midpoints, and centroid of the simplex."""
    annotations = [
        (r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)", (1/3, 1/3), (1/3, .23)),
        (r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)", (.5, .0), (.5, .1)),
        (r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)", (.0, .5), (.1, .5)),
        (r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)", (.5, .5), (.6, .6)),
        (r"($0$, $0$, $1$)", (0, 0), (.1, .1)),
        (r"($1$, $0$, $0$)", (1, 0), (1, .1)),
        (r"($0$, $1$, $0$)", (0, 1), (.1, 1)),
    ]
    for text, xy, xytext in annotations:
        ax.annotate(text, xy=xy, xytext=xytext, xycoords="data",
                    arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center", verticalalignment="center")
    ax.plot([1/3], [1/3], "ko", ms=5)
def style_simplex_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Probability class 1")
    ax.set_ylabel("Probability class 2")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(False)
# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_probability_shift(y_test, clf_probs, sig_clf_probs):
    fig, ax = plt.subplots()
    for i in range(clf_probs.shape[0]):
        ax.arrow(clf_probs[i, 0], clf_probs[i, 1],
                  sig_clf_probs[i, 0] - clf_probs[i, 0],
                  sig_clf_probs[i, 1] - clf_probs[i, 1],
                  color=COLORS[y_test[i]], head_width=1e-2)
    ax.plot([1.0], [0.0], "ro", ms=20, label="Class 1")
    ax.plot([0.0], [1.0], "go", ms=20, label="Class 2")
    ax.plot([0.0], [0.0], "bo", ms=20, label="Class 3")
    draw_simplex_grid(ax)
    annotate_simplex_vertices(ax)
    style_simplex_axes(ax, "Change of predicted probabilities after sigmoid calibration")
    ax.legend(loc="best")
    return fig
def plot_calibrator_map(sig_clf):
    fig, ax = plt.subplots()
    # Grid over the 2-simplex.
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
    p = p[p[:, 2] >= 0]
    calibrated_classifier = sig_clf.calibrated_classifiers_[0]
    prediction = np.vstack([
        calibrator.predict(this_p)
        for calibrator, this_p in zip(calibrated_classifier.calibrators_, p.T)
    ]).T
    prediction = safe_normalize(prediction)
    for i in range(prediction.shape[0]):
        ax.arrow(p[i, 0], p[i, 1],
                  prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
                  head_width=1e-2, color=COLORS[np.argmax(p[i])])
    draw_simplex_grid(ax)
    style_simplex_axes(ax, "Illustration of sigmoid calibrator")
    return fig
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    data = make_dataset()
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_train_valid, y_train_valid = data["train_valid"]
    X_test, y_test = data["test"]
    # Uncalibrated: trained on train+valid, evaluated on test.
    clf_uncal = fit_uncalibrated(X_train_valid, y_train_valid)
    uncal_probs = clf_uncal.predict_proba(X_test)
    uncal_score = log_loss(y_test, uncal_probs)
    # Calibrated: trained on train, calibrated on valid, evaluated on test.
    clf, sig_clf = fit_calibrated(X_train, y_train, X_valid, y_valid)
    clf_probs = clf.predict_proba(X_test)
    sig_clf_probs = sig_clf.predict_proba(X_test)
    sig_score = log_loss(y_test, sig_clf_probs)
    print("Log-loss of")
    print(f" * uncalibrated classifier trained on 800 datapoints: {uncal_score:.3f}")
    print(f" * classifier trained on 600 datapoints and calibrated on "
          f"200 datapoints: {sig_score:.3f}")
    plot_probability_shift(y_test, clf_probs, sig_clf_probs)
    plot_calibrator_map(sig_clf)
    plt.show()
if __name__ == "__main__":
    main()
