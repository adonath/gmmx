# Density Estimation of Moon Data. This exampled is adapted from "In Depth: Gaussian Mixture Models" chapter of
# the Python Data Science Handbook by Jake VanderPlas. The original code can be found
# at https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.datasets import make_moons

from gmmx import EMFitter, GaussianMixtureModelJax


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(
                xy=position,
                width=nsig * width,
                height=nsig * height,
                angle=angle,
                **kwargs,
            )
        )


def plot_gmm(gmm, X, label=True, ax=None):
    """Plot the GMM"""
    ax = ax or plt.gca()

    labels = gmm.predict(X)

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="viridis", zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
    ax.axis("equal")

    w_factor = 0.2 / gmm.weights_numpy.max()
    for pos, covar, w in zip(
        gmm.means_numpy, gmm.covariances.values_numpy, gmm.weights_numpy
    ):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax)


def fit_and_plot_gmm(n_components, ax=None):
    """Fit and plot a GMM"""
    ax = ax or plt.gca()
    x, _ = make_moons(200, noise=0.05, random_state=0)
    ax.scatter(x[:, 0], x[:, 1])
    ax.text(
        0.95,
        0.9,
        f"N Components: {n_components}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    gmm = GaussianMixtureModelJax.from_k_means(x, n_components=n_components)

    fitter = EMFitter(tol=1e-4, max_iter=100)
    result = fitter.fit(x=x, gmm=gmm)

    plot_gmm(result.gmm, x, ax=ax)
    return ax


if __name__ == "__main__":
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))

    for idx, ax in enumerate(axes.flat):
        ax = fit_and_plot_gmm(idx + 1, ax=ax)

    plt.tight_layout()
    plt.show()
