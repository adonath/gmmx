# A simple example of fitting a Gaussian Mixture Model (GMM) to a 1D dataset.

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from gmmx import EMFitter, GaussianMixtureModelJax


def plot_gmm(ax, weights, means, covariances, **kwargs):
    """Plot a GMM"""
    x = np.linspace(-5, 15, 1000)
    y = np.zeros_like(x)

    for w, m, c in zip(weights, means, covariances):
        y += w * norm.pdf(x, m, np.sqrt(c))

    ax.plot(x, y, **kwargs)


def fit_and_plot_jax(ax, x, **kwargs):
    """Fit and plot Jax GMM"""
    gmm = GaussianMixtureModelJax.from_squeezed(
        means=np.array([[1.0], [8.0]]),
        covariances=np.array([[[1.0]], [[1.0]]]),
        weights=np.array([0.5, 0.5]),
    )

    fitter = EMFitter(max_iter=100, tol=0.1)
    result = fitter.fit(x=x, gmm=gmm)

    plot_gmm(
        ax=ax,
        weights=result.gmm.weights.flatten(),
        means=result.gmm.means.flatten(),
        covariances=result.gmm.covariances.values.flatten(),
        **kwargs,
    )


def fit_and_plot_sklearn(ax, x, **kwargs):
    """Fit and plot sklearn GMM"""
    gmm_sk = GaussianMixture(
        n_components=2,
        max_iter=100,
        tol=0.1,
        means_init=np.array([[1.0], [8.0]]),
        weights_init=np.array([0.5, 0.5]),
        precisions_init=np.array([[[1.0]], [[1.0]]]),
    )
    gmm_sk.fit(x)

    plot_gmm(
        ax=ax,
        weights=gmm_sk.weights_,
        means=gmm_sk.means_.flatten(),
        covariances=gmm_sk.covariances_.flatten(),
        **kwargs,
    )


if __name__ == "__main__":
    gmm_jax = GaussianMixtureModelJax.from_squeezed(
        means=np.array([[0], [10]]),
        covariances=np.array([[[2]], [[1]]]),
        weights=np.array([0.2, 0.8]),
    )

    n_samples = 100_000

    key = jax.random.PRNGKey(0)
    x, _ = gmm_jax.sample(key, n_samples=n_samples)

    ax = plt.subplot(111)
    ax.hist(x, bins=100, density=True)
    fit_and_plot_jax(ax=ax, x=x, label="gmmx", ls="-")
    fit_and_plot_sklearn(ax=ax, x=x, label="sklearn", ls="--")
    ax.set_ylabel("PDF (normalized)")
    ax.set_xlabel("x")

    plt.legend()
    plt.show()
