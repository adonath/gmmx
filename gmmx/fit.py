from dataclasses import dataclass

import jax
from jax import numpy as jnp

from .gmm import Axis, GaussianMixtureModelJax
from .utils import register_dataclass_jax

__all__ = ["EMFitter", "EMFitterResult"]


@register_dataclass_jax(
    data_fields=["x", "gmm", "n_iter", "log_likelihood", "log_likelihood_diff"],
)
@dataclass
class EMFitterResult:
    """Expectation-Maximization Fitter Result

    Attributes
    ----------
    x : jax.array
        Feature vectors
    gmm : GaussianMixtureModelJax
        Gaussian mixture model instance.
    n_iter : int
        Number of iterations
    log_likelihood : jax.array
        Log-likelihood of the data
    log_likelihood_diff : jax.array
        Difference in log-likelihood with respect to the previous iteration
    """

    x: jax.Array
    gmm: GaussianMixtureModelJax
    n_iter: int
    log_likelihood: jax.Array
    log_likelihood_diff: jax.Array


@register_dataclass_jax(meta_fields=["max_iter", "tol", "reg_covar"])
@dataclass
class EMFitter:
    """Expectation-Maximization Fitter

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance
    reg_covar : float
        Regularization for covariance matrix
    """

    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-6

    def e_step(self, x, gmm):
        """Expectation step

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        log_likelihood : jax.array
            Log-likelihood of the data
        """
        log_prob = gmm.estimate_log_prob(x)
        log_prob_norm = jax.scipy.special.logsumexp(
            log_prob, axis=Axis.components, keepdims=True
        )
        log_resp = log_prob - log_prob_norm
        return jnp.mean(log_prob_norm), log_resp

    def m_step(self, x, gmm, log_resp):
        """Maximization step

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        log_resp : jax.array
            Logarithm of the responsibilities

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Updated Gaussian mixture model instance.
        """
        xp = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))

        resp = jnp.exp(log_resp)
        nk = jnp.sum(resp, axis=Axis.batch, keepdims=True)
        means = jnp.matmul(resp.T, xp.T.mT).T / nk
        covariances = gmm.covariances.estimate(
            x=xp, means=means, resp=resp, nk=nk, reg_covar=self.reg_covar
        )
        return gmm.__class__(
            weights=nk / nk.sum(), means=means, covariances=covariances
        )

    @jax.jit
    def fit(self, x, gmm):
        """Fit the model to the data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        result : EMFitterResult
            Fitting result
        """

        def em_step(args):
            x, gmm, n_iter, log_likelihood_prev, _ = args
            log_likelihood, log_resp = self.e_step(x, gmm)
            gmm = self.m_step(x, gmm, log_resp)
            return (
                x,
                gmm,
                n_iter + 1,
                log_likelihood,
                jnp.abs(log_likelihood - log_likelihood_prev),
            )

        def em_cond(args):
            _, _, n_iter, _, log_likelihood_diff = args
            return (n_iter < self.max_iter) & (log_likelihood_diff > self.tol)

        result = jax.lax.while_loop(
            cond_fun=em_cond,
            body_fun=em_step,
            init_val=(x, gmm, 0, jnp.asarray(1e25), jnp.array(jnp.inf)),
        )
        return EMFitterResult(*result)