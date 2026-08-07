from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import optax  # type: ignore [import-untyped]
from jax import numpy as jnp

from .gmm import COVARIANCE, Axis, CovarianceType, GaussianMixtureModelJax
from .utils import register_dataclass_jax

# optax does not ship type information, so we alias the types we use
GradientTransformation = Any
OptState = Any

__all__ = [
    "AdamFitter",
    "AdamFitterResult",
    "EMFitter",
    "EMFitterResult",
    "UnconstrainedParameters",
]


@dataclass
class FitterResult:
    """Fitter Result

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
    converged : bool
        Whether the algorithm converged
    """

    x: jax.Array
    gmm: GaussianMixtureModelJax
    n_iter: int
    log_likelihood: jax.Array
    log_likelihood_diff: jax.Array
    converged: bool


@register_dataclass_jax(
    data_fields=[
        "x",
        "gmm",
        "n_iter",
        "log_likelihood",
        "log_likelihood_diff",
        "converged",
    ]
)
@dataclass
class EMFitterResult(FitterResult):
    """Expectation-Maximization Fitter Result"""


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

    def e_step(
        self, x: jax.Array, gmm: GaussianMixtureModelJax
    ) -> tuple[jax.Array, jax.Array]:
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
        log_prob = gmm.log_prob(x)
        log_prob_norm = jax.scipy.special.logsumexp(
            log_prob, axis=Axis.components, keepdims=True
        )
        log_resp = log_prob - log_prob_norm
        return jnp.mean(log_prob_norm), log_resp

    def m_step(
        self, x: jax.Array, gmm: GaussianMixtureModelJax, log_resp: jax.Array
    ) -> GaussianMixtureModelJax:
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
        x = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
        return gmm.from_responsibilities(
            x,
            jnp.exp(log_resp),
            reg_covar=self.reg_covar,
            covariance_type=gmm.covariances.type,
        )

    @jax.jit
    def fit(self, x: jax.Array, gmm: GaussianMixtureModelJax) -> EMFitterResult:
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

        def em_step(
            args: tuple[jax.Array, GaussianMixtureModelJax, int, jax.Array, jax.Array],
        ) -> tuple:
            """EM step function"""
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

        def em_cond(
            args: tuple[jax.Array, GaussianMixtureModelJax, int, jax.Array, jax.Array],
        ) -> jax.Array:
            """EM stop condition function"""
            _, _, n_iter, _, log_likelihood_diff = args
            return (n_iter < self.max_iter) & (log_likelihood_diff >= self.tol)

        result = jax.lax.while_loop(
            cond_fun=em_cond,
            body_fun=em_step,
            init_val=(x, gmm, 0, jnp.asarray(jnp.inf), jnp.array(jnp.inf)),
        )
        return EMFitterResult(*result, converged=result[2] < self.max_iter)


@register_dataclass_jax(
    data_fields=["weights", "means", "covariances"], meta_fields=["covariance_type"]
)
@dataclass
class UnconstrainedParameters:
    """Unconstrained parametrization of a Gaussian mixture model

    Gradient based optimizers require parameters that are free to take any real
    value, while the weights of a Gaussian mixture model are constrained to a
    simplex and the covariances to be positive definite. Here the weights are
    represented by their logits and the covariances by the covariance type
    specific unconstrained representation.

    Attributes
    ----------
    weights : jax.array
        Logits of the weights. Expected shape is (1, n_components, 1, 1)
    means : jax.array
        Mean of each component. Expected shape is (1, n_components, n_features, 1)
    covariances : jax.array
        Unconstrained covariance values, see e.g. `~gmmx.gmm.FullCovariances.from_unconstrained`
    covariance_type : CovarianceType
        Covariance type
    """

    weights: jax.Array
    means: jax.Array
    covariances: jax.Array
    covariance_type: CovarianceType = CovarianceType.full

    @classmethod
    def from_gmm(cls, gmm: GaussianMixtureModelJax) -> UnconstrainedParameters:
        """Create the unconstrained parameters from a Gaussian mixture model

        Parameters
        ----------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        parameters : UnconstrainedParameters
            Unconstrained parameters instance.
        """
        return cls(
            weights=gmm.log_weights,
            means=gmm.means,
            covariances=gmm.covariances.values_unconstrained,
            covariance_type=gmm.covariances.type,
        )

    def to_gmm(self) -> GaussianMixtureModelJax:
        """Convert the unconstrained parameters back into a Gaussian mixture model

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariances = COVARIANCE[self.covariance_type].from_unconstrained(
            self.covariances
        )
        return GaussianMixtureModelJax(
            weights=jax.nn.softmax(self.weights, axis=Axis.components),
            means=self.means,
            covariances=covariances,
        )


@register_dataclass_jax(
    data_fields=[
        "x",
        "gmm",
        "n_iter",
        "log_likelihood",
        "log_likelihood_diff",
        "converged",
    ]
)
@dataclass
class AdamFitterResult(FitterResult):
    """Adam Fitter Result"""


@register_dataclass_jax(
    meta_fields=["max_iter", "tol", "learning_rate", "beta_1", "beta_2"]
)
@dataclass
class AdamFitter:
    """Adam Fitter

    Maximizes the log-likelihood of the data by gradient descent using the Adam
    optimizer from `~optax.adam`. Compared to the `~EMFitter` this typically
    requires many more iterations, but in exchange it does not rely on the
    closed form solution of the maximization step.

    The model is optimized in the unconstrained parametrization defined by
    `~UnconstrainedParameters`, which maps any real valued parameter vector to
    valid weights and positive definite covariances. This is why, unlike for
    the `~EMFitter`, no regularization of the covariances is required.

    The convergence criterion is the same as for the `~EMFitter`: the fit stops
    once the change in log-likelihood between two subsequent iterations drops
    below `tol`. As the steps taken are much smaller than for the EM algorithm,
    the tolerance should be chosen smaller as well.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance
    learning_rate : float
        Learning rate
    beta_1 : float
        Exponential decay rate for the first moment estimates
    beta_2 : float
        Exponential decay rate for the second moment estimates
    """

    max_iter: int = 1_000
    tol: float = 1e-6
    learning_rate: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999

    @property
    def optimizer(self) -> GradientTransformation:
        """Optimizer (`~optax.GradientTransformation`)"""
        return optax.adam(
            learning_rate=self.learning_rate, b1=self.beta_1, b2=self.beta_2
        )

    def loss(self, parameters: UnconstrainedParameters, x: jax.Array) -> jax.Array:
        """Negative log-likelihood of the data

        Parameters
        ----------
        parameters : UnconstrainedParameters
            Unconstrained parameters instance.
        x : jax.array
            Feature vectors

        Returns
        -------
        loss : jax.array
            Negative log-likelihood of the data
        """
        return -parameters.to_gmm().score(x)  # type: ignore [no-any-return]

    def step(
        self, x: jax.Array, parameters: UnconstrainedParameters, state: OptState
    ) -> tuple[jax.Array, UnconstrainedParameters, OptState]:
        """Gradient descent step

        Parameters
        ----------
        x : jax.array
            Feature vectors
        parameters : UnconstrainedParameters
            Unconstrained parameters instance.
        state : optax.OptState
            State of the optimizer

        Returns
        -------
        log_likelihood, parameters, state : tuple
            Log-likelihood of the data, updated parameters and updated optimizer state
        """
        loss, gradients = jax.value_and_grad(self.loss)(parameters, x)
        updates, state = self.optimizer.update(gradients, state, parameters)
        return -loss, optax.apply_updates(parameters, updates), state

    @jax.jit
    def fit(self, x: jax.Array, gmm: GaussianMixtureModelJax) -> AdamFitterResult:
        """Fit the model to the data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        result : AdamFitterResult
            Fitting result
        """
        parameters = UnconstrainedParameters.from_gmm(gmm)

        def adam_step(
            args: tuple[
                jax.Array,
                UnconstrainedParameters,
                int,
                jax.Array,
                jax.Array,
                OptState,
            ],
        ) -> tuple:
            """Adam step function"""
            x, parameters, n_iter, log_likelihood_prev, _, state = args
            log_likelihood, parameters, state = self.step(x, parameters, state)
            return (
                x,
                parameters,
                n_iter + 1,
                log_likelihood,
                jnp.abs(log_likelihood - log_likelihood_prev),
                state,
            )

        def adam_cond(
            args: tuple[
                jax.Array,
                UnconstrainedParameters,
                int,
                jax.Array,
                jax.Array,
                OptState,
            ],
        ) -> jax.Array:
            """Adam stop condition function"""
            _, _, n_iter, _, log_likelihood_diff, _ = args
            return (n_iter < self.max_iter) & (log_likelihood_diff >= self.tol)

        result = jax.lax.while_loop(
            cond_fun=adam_cond,
            body_fun=adam_step,
            init_val=(
                x,
                parameters,
                0,
                jnp.asarray(jnp.inf),
                jnp.array(jnp.inf),
                self.optimizer.init(parameters),
            ),
        )
        x, parameters, n_iter, log_likelihood, log_likelihood_diff, _ = result
        return AdamFitterResult(
            x=x,
            gmm=parameters.to_gmm(),
            n_iter=n_iter,
            log_likelihood=log_likelihood,
            log_likelihood_diff=log_likelihood_diff,
            converged=n_iter < self.max_iter,
        )
