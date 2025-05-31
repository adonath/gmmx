"""Some notes on the implementation:

I have not tried to keep the implementation close to the sklearn implementation.
I have rather tried to realize my own best practices for code structure and
clarity. Here are some more detailed thoughts:

1. **Use dataclasses for the model representation**: this reduces the amount of
boilerplate code for initialization and in combination with the `register_dataclass_jax`
decorator it integrates seamleassly with JAX.

2. **Split up the different covariance types into different classes**: this avoids
the need for multiple blocks of if-else statements.

3. **Use a registry for the covariance types**:  This allows for easy extensibility
by the user.

3. **Remove Python loops**: I have not checked the reason why the sklearn implementation
still uses Python loops, but my guess is that it is simpler(?) and when there are
operations such as matmul and cholesky decomposition, the Python loop does not become
the bottleneck. In JAX, however, it is usually better to avoid Python loops and let
the JAX compiler take care of the optimization instead.

4. **Rely on same internal array dimension and axis order**:
Internally all(!) involved arrays (even 1d weights) are represented as 4d arrays
with the axes (batch, components, features, features_covar). This makes it much
easier to write array operations and rely on broadcasting. This minimizes the
amount of in-line reshaping and in-line extension of dimensions. If you think
about it, this is most likely the way how array programming was meant to be used
in first place. Yet, I have rarely seen this in practice, probably because people
struggle with the additional dimensions in the beginning. However once you get
used to it, it is much easier to write and understand the code! The only downside
is that the user has to face the additional "empty" dimensions when directy working
with the arrays. For convenience I have introduced properties, that return the arrays
with the empty dimensions removed. Another downside maybe that you have to use `keepdims=True`
more often, but there I would even argue that the default behavior in the array libraries
should change.

5. **"Poor-peoples" named axes**: The axis order convention is defined in the
code in the `Axis` enum, which maps the name to the integer dimension. Later I
can use, e.g. `Axis.batch` to refer to the batch axis in the code. This is the
simplest way to come close to named axes in any array library! So you can use
e.g. `jnp.sum(x, axes=Axis.components)` to sum over the components axis. I found
this to be a very powerful concept that improves the code clarity a lot, yet I
have not seen it often in other libraries. Of course there is `einops` but the
simple enum works just fine in many cases!

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, ClassVar, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp

from gmmx.utils import register_dataclass_jax

__all__ = [
    "Axis",
    "CovarianceType",
    "DiagCovariances",
    "FullCovariances",
    "GaussianMixtureModelJax",
    "GaussianMixtureSKLearn",
]


# AnyArray = Union[np.typing.NDArray, jax.Array]
Device = Union[str, None]


class CovarianceType(str, Enum):
    """Convariance type"""

    full = "full"
    diag = "diag"


class Axis(int, Enum):
    """Internal axis order"""

    batch = 0
    components = 1
    features = 2
    features_covar = 3


def check_shape(array: jax.Array, expected: tuple[int | None, ...]) -> None:
    """Check shape of array"""
    if array.dtype != jnp.float32:
        message = f"Expected float32, got {array.dtype}"
        raise ValueError(message)

    if len(array.shape) != len(expected):
        message = f"Expected shape {expected}, got {array.shape}"
        raise ValueError(message)

    for n, m in zip(array.shape, expected):
        if m is not None and n != m:
            message = f"Expected shape {expected}, got {array.shape}"
            raise ValueError(message)


@register_dataclass_jax(data_fields=["values"])
@dataclass
class FullCovariances:
    """Full covariance matrix

    Attributes
    ----------
    values : jax.array
        Covariance values. Expected shape is (1, n_components, n_features, n_features)
    """

    values: jax.Array
    type: ClassVar[CovarianceType] = CovarianceType.full

    def __post_init__(self) -> None:
        check_shape(self.values, (1, None, None, None))

    @classmethod
    def from_squeezed(cls, values) -> FullCovariances:
        """Create a covariance matrix from squeezed array

        Parameters
        ----------
        values : jax.Array ot np.array
            Covariance values. Expected shape is (n_components, n_features, n_features)

        Returns
        -------
        covariances : FullCovariances
            Covariance matrix instance.
        """
        if values.ndim != 3:
            message = f"Expected array of shape (n_components, n_features, n_features), got {values.shape}"
            raise ValueError(message)

        return cls(values=jnp.expand_dims(values, axis=Axis.batch))

    @property
    def values_numpy(self):
        """Covariance as numpy array"""
        return np.squeeze(np.asarray(self.values), axis=Axis.batch)

    @property
    def values_dense(self) -> jax.Array:
        """Covariance as dense matrix"""
        return self.values

    @property
    def precisions_cholesky_numpy(self):
        """Compute precision matrices"""
        return np.squeeze(np.asarray(self.precisions_cholesky), axis=Axis.batch)

    @classmethod
    def create(
        cls, n_components: int, n_features: int, device: Device = None
    ) -> FullCovariances:
        """Create covariance matrix

        By default the covariance matrix is set to the identity matrix.

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features
        device : str, optional
            Device, by default None

        Returns
        -------
        covariances : FullCovariances
            Covariance matrix instance.
        """
        identity = jnp.expand_dims(
            jnp.eye(n_features), axis=(Axis.batch, Axis.components)
        )

        values = jnp.repeat(identity, n_components, axis=Axis.components)
        values = jax.device_put(values, device=device)
        return cls(values=values)

    def log_prob(self, x: jax.Array, means: jax.Array) -> jax.Array:
        """Compute log likelihood from the covariance for a given feature vector

        Parameters
        ----------
        x : jax.array
            Feature vectors
        means : jax.array
            Means of the components

        Returns
        -------
        log_prob : jax.array
            Log likelihood
        """
        precisions_cholesky = self.precisions_cholesky

        y = jnp.matmul(x.mT, precisions_cholesky) - jnp.matmul(
            means.mT, precisions_cholesky
        )
        return jnp.sum(
            jnp.square(y),
            axis=(Axis.features, Axis.features_covar),
            keepdims=True,
        )

    @classmethod
    def from_responsibilities(
        cls,
        x: jax.Array,
        means: jax.Array,
        resp: jax.Array,
        nk: jax.Array,
        reg_covar: float,
    ) -> FullCovariances:
        """Estimate updated covariance matrix from data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        means : jax.array
            Means of the components
        resp : jax.array
            Responsibilities
        nk : jax.array
            Number of samples in each component
        reg_covar : float
            Regularization for the covariance matrix

        Returns
        -------
        covariances : FullCovariances
            Updated covariance matrix instance.
        """
        diff = x - means
        axes = (Axis.features_covar, Axis.components, Axis.features, Axis.batch)
        diff = jnp.transpose(diff, axes=axes)
        resp = jnp.transpose(resp, axes=axes)
        values = jnp.matmul(resp * diff, diff.mT) / nk
        idx = jnp.arange(x.shape[Axis.features])
        values = values.at[:, :, idx, idx].add(reg_covar)
        return cls(values=values)

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.values.shape[Axis.components]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.values.shape[Axis.features]

    @property
    def n_parameters(self) -> int:
        """Number of parameters"""
        return int(self.n_components * self.n_features * (self.n_features + 1) / 2.0)

    @property
    def log_det_cholesky(self) -> jax.Array:
        """Log determinant of the cholesky decomposition"""
        diag = jnp.trace(
            jnp.log(self.precisions_cholesky),
            axis1=Axis.features,
            axis2=Axis.features_covar,
        )
        return jnp.expand_dims(diag, axis=(Axis.features, Axis.features_covar))

    @property
    def precisions_cholesky(self) -> jax.Array:
        """Compute precision matrices"""
        cov_chol = jsp.linalg.cholesky(self.values, lower=True)

        identity = jnp.expand_dims(
            jnp.eye(self.n_features), axis=(Axis.batch, Axis.components)
        )

        b = jnp.repeat(identity, self.n_components, axis=Axis.components)
        precisions_chol = jsp.linalg.solve_triangular(cov_chol, b, lower=True)
        return precisions_chol.mT

    @classmethod
    def from_precisions(cls, precisions) -> FullCovariances:
        """Create covariance matrix from precision matrices"""
        values = jsp.linalg.inv(precisions)
        return cls.from_squeezed(values=values)


@register_dataclass_jax(data_fields=["values"])
@dataclass
class DiagCovariances:
    """Diagonal covariance matrices"""

    values: jax.Array
    type: ClassVar[CovarianceType] = CovarianceType.diag

    def __post_init__(self) -> None:
        check_shape(self.values, (1, None, None, 1))

    @property
    def values_dense(self) -> jax.Array:
        """Covariance as dense matrix"""
        values = jnp.zeros((1, self.n_components, self.n_features, self.n_features))
        idx = jnp.arange(self.n_features)
        covar_diag = jnp.squeeze(self.values, axis=(Axis.batch, Axis.features_covar))
        return values.at[:, :, idx, idx].set(covar_diag)

    @classmethod
    def from_squeezed(cls, values) -> DiagCovariances:
        """Create a diagonal covariance matrix from squeezed array

        Parameters
        ----------
        values : jax.Array ot np.array
            Covariance values. Expected shape is (n_components, n_features)

        Returns
        -------
        covariances : FullCovariances
            Covariance matrix instance.
        """
        if values.ndim != 2:
            message = f"Expected array of shape (n_components, n_features), got {values.shape}"
            raise ValueError(message)

        return cls(
            values=jnp.expand_dims(values, axis=(Axis.batch, Axis.features_covar))
        )

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.values.shape[Axis.components]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.values.shape[Axis.features]

    @property
    def n_parameters(self) -> int:
        """Number of parameters"""
        return int(self.n_components * self.n_features)

    @classmethod
    def from_responsibilities(
        cls,
        x: jax.Array,
        means: jax.Array,
        resp: jax.Array,
        nk: jax.Array,
        reg_covar: float,
    ) -> DiagCovariances:
        """Estimate updated covariance matrix from data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        means : jax.array
            Means of the components
        resp : jax.array
            Responsibilities
        nk : jax.array
            Number of samples in each component
        reg_covar : float
            Regularization for the covariance matrix

        Returns
        -------
        covariances : FullCovariances
            Updated covariance matrix instance.
        """
        x_squared_mean = jnp.sum(resp * x**2, axis=Axis.batch, keepdims=True) / nk
        values = x_squared_mean - means**2 + reg_covar
        return cls(values=values)

    @property
    def precisions_cholesky_sparse(self) -> jax.Array:
        """Compute precision matrices"""
        return jnp.sqrt(1.0 / self.values).mT

    @property
    def precisions_cholesky_numpy(self):
        """Compute precision matrices"""
        return np.squeeze(
            np.asarray(self.precisions_cholesky_sparse),
            axis=(Axis.batch, Axis.features),
        )

    @property
    def values_numpy(self):
        """Covariance as numpy array"""
        return np.squeeze(
            np.asarray(self.values), axis=(Axis.batch, Axis.features_covar)
        )

    @property
    def log_det_cholesky(self) -> jax.Array:
        """Log determinant of the cholesky decomposition"""
        return jnp.sum(
            jnp.log(self.precisions_cholesky_sparse),
            axis=(Axis.features, Axis.features_covar),
            keepdims=True,
        )

    def log_prob(self, x: jax.Array, means: jax.Array) -> jax.Array:
        """Compute log likelihood from the covariance for a given feature vector"""
        precisions_cholesky = self.precisions_cholesky_sparse
        y = (x.mT * precisions_cholesky) - (means.mT * precisions_cholesky)
        return jnp.sum(
            jnp.square(y),
            axis=(Axis.features, Axis.features_covar),
            keepdims=True,
        )

    @classmethod
    def from_precisions(cls, precisions) -> DiagCovariances:
        """Create covariance matrix from precision matrices"""
        values = 1.0 / precisions
        return cls.from_squeezed(values=values)


COVARIANCE: dict[CovarianceType, Any] = {
    FullCovariances.type: FullCovariances,
    DiagCovariances.type: DiagCovariances,
}

# keep this mapping separate, as names in sklearn might change
SKLEARN_COVARIANCE_TYPE: dict[Any, str] = {
    FullCovariances: "full",
    DiagCovariances: "diag",
}


@register_dataclass_jax(data_fields=["weights", "means", "covariances"])
@dataclass
class GaussianMixtureModelJax:
    """Gaussian Mixture Model

    Attributes
    ----------
    weights : jax.array
        Weights of each component. Expected shape is (1, n_components, 1, 1)
    means : jax.array
        Mean of each component. Expected shape is (1, n_components, n_features, 1)
    covariances : jax.array
        Covariance of each component. Expected shape is (1, n_components, n_features, n_features)
    """

    weights: jax.Array
    means: jax.Array
    covariances: FullCovariances

    def __post_init__(self) -> None:
        check_shape(self.weights, (1, None, 1, 1))
        check_shape(self.means, (1, None, None, 1))

    @property
    def weights_numpy(self):
        """Weights as numpy array"""
        return np.squeeze(
            np.asarray(self.weights),
            axis=(Axis.batch, Axis.features, Axis.features_covar),
        )

    @property
    def means_numpy(self):
        """Means as numpy array"""
        return np.squeeze(
            np.asarray(self.means), axis=(Axis.batch, Axis.features_covar)
        )

    @classmethod
    def create(
        cls,
        n_components: int,
        n_features: int,
        covariance_type: CovarianceType = CovarianceType.full,
        device: Device = None,
    ) -> GaussianMixtureModelJax:
        """Create a GMM from configuration

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features
        covariance_type : str, optional
            Covariance type, by default "full"
        device : str, optional
            Device, by default None

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        weights = jnp.ones((1, n_components, 1, 1)) / n_components
        means = jnp.zeros((1, n_components, n_features, 1))
        covariances = COVARIANCE[covariance_type].create(
            n_components, n_features, device=device
        )
        return cls(
            weights=jax.device_put(weights, device=device),
            means=jax.device_put(means, device=device),
            covariances=covariances,
        )

    @classmethod
    def from_squeezed(
        cls,
        means,
        covariances,
        weights,
        covariance_type: CovarianceType | str = CovarianceType.full,
    ) -> GaussianMixtureModelJax:
        """Create a Jax GMM from squeezed arrays

        Parameters
        ----------
        means : jax.Array or np.array
            Mean of each component. Expected shape is (n_components, n_features)
        covariances : jax.Array or np.array
            Covariance of each component. Expected shape is (n_components, n_features, n_features)
        weights : jax.Array or np.array
            Weights of each component. Expected shape is (n_components,)
        covariance_type : str, optional
            Covariance type, by default "full"

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        means = jnp.expand_dims(means, axis=(Axis.batch, Axis.features_covar))
        weights = jnp.expand_dims(
            weights, axis=(Axis.batch, Axis.features, Axis.features_covar)
        )

        covariances = COVARIANCE[covariance_type].from_squeezed(values=covariances)
        return cls(weights=weights, means=means, covariances=covariances)  # type: ignore [arg-type]

    @classmethod
    def from_responsibilities(
        cls,
        x: jax.Array,
        resp: jax.Array,
        reg_covar: float,
        covariance_type: CovarianceType = CovarianceType.full,
    ) -> GaussianMixtureModelJax:
        """Update parameters

        Parameters
        ----------
        x : jax.array
            Feature vectors
        resp : jax.array
            Responsibilities
        reg_covar : float
            Regularization for the covariance matrix
        covariance_type : str, optional
            Covariance type, by default "full"

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Updated Gaussian mixture model
        """
        covariance_type = CovarianceType(covariance_type)

        # I don't like the hard-coded 10 here, but it is the same as in sklearn
        nk = (
            jnp.sum(resp, axis=Axis.batch, keepdims=True)
            + 10 * jnp.finfo(resp.dtype).eps
        )
        means = jnp.matmul(resp.T, x.T.mT).T / nk
        covariances = COVARIANCE[covariance_type].from_responsibilities(
            x=x, means=means, resp=resp, nk=nk, reg_covar=reg_covar
        )
        return cls(weights=nk / nk.sum(), means=means, covariances=covariances)

    @classmethod
    def from_k_means(
        cls,
        x,
        n_components: int,
        reg_covar: float = 1e-6,
        covariance_type: CovarianceType = CovarianceType.full,
        **kwargs: dict,
    ) -> GaussianMixtureModelJax:
        """Init from k-means clustering

        Parameters
        ----------
        x : jax.array
            Feature vectors
        n_components : int
            Number of components
        reg_covar : float, optional
            Regularization for the covariance matrix, by default 1e6
        covariance_type : str, optional
            Covariance type, by default "full"
        **kwargs : dict
            Additional arguments passed to `~sklearn.cluster.KMeans`

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        from sklearn.cluster import KMeans  # type: ignore [import-untyped]

        n_samples = x.shape[Axis.batch]

        resp = jnp.zeros((n_samples, n_components))

        kwargs.setdefault("n_init", 10)  # type: ignore [arg-type]
        label = KMeans(n_clusters=n_components, **kwargs).fit(x).labels_

        idx = jnp.arange(n_samples)
        resp = resp.at[idx, label].set(1.0)

        xp = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
        resp = jnp.expand_dims(resp, axis=(Axis.features, Axis.features_covar))
        return cls.from_responsibilities(
            xp, resp, reg_covar=reg_covar, covariance_type=covariance_type
        )

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.covariances.n_features

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.covariances.n_components

    @property
    def n_parameters(self) -> int:
        """Number of parameters"""
        return int(
            self.n_components
            + self.n_components * self.n_features
            + self.covariances.n_parameters
            - 1
        )

    @property
    def log_weights(self) -> jax.Array:
        """Log weights (~jax.ndarray)"""
        return jnp.log(self.weights)

    @jax.jit
    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log likelihood for given feature vector

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        log_prob : jax.array
            Log likelihood
        """
        x = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
        log_prob = self.covariances.log_prob(x, self.means)
        two_pi = jnp.array(2 * jnp.pi)

        value = (
            -0.5 * (self.n_features * jnp.log(two_pi) + log_prob)
            + self.covariances.log_det_cholesky
            + self.log_weights
        )
        return value

    def to_sklearn(self, **kwargs: dict[str, Any]) -> Any:
        """Convert to sklearn GaussianMixture

        The methods sets the weights, means, precisions_cholesky and covariances_ attributes,
        however sklearn will overvwrite them when fitting the model.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to `~sklearn.mixture.GaussianMixture`

        Returns
        -------
        gmm : `~sklearn.mixture.GaussianMixture`
            Gaussian mixture model instance.
        """
        from sklearn.mixture import GaussianMixture  # type: ignore [import-untyped]

        kwargs.setdefault("warm_start", True)  # type: ignore [arg-type]
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=SKLEARN_COVARIANCE_TYPE[type(self.covariances)],
            **kwargs,
        )
        # This does a warm start at the given parameters
        gmm.converged_ = True
        gmm.lower_bound_ = -np.inf
        gmm.weights_ = self.weights_numpy
        gmm.means_ = self.means_numpy
        gmm.precisions_cholesky_ = self.covariances.precisions_cholesky_numpy
        gmm.covariances_ = self.covariances.values_numpy
        return gmm

    @jax.jit
    def predict(self, x: jax.Array) -> jax.Array:
        """Predict the component index for each sample

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        predictions : jax.array
            Predicted component index
        """
        log_prob = self.log_prob(x)
        predictions = jnp.argmax(log_prob, axis=Axis.components, keepdims=True)
        return jnp.squeeze(predictions, axis=(Axis.features, Axis.features_covar))

    @jax.jit
    def predict_proba(self, x: jax.Array) -> jax.Array:
        """Predict the probability of each sample belonging to each component

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        probabilities : jax.array
            Predicted probabilities
        """
        log_prob = self.log_prob(x)
        log_prob_norm = jax.scipy.special.logsumexp(
            log_prob, axis=Axis.components, keepdims=True
        )
        return jnp.exp(log_prob - log_prob_norm)

    @jax.jit
    def score_samples(self, x: jax.Array) -> jax.Array:
        """Compute the weighted log probabilities for each sample

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        log_prob : jax.array
            Log probabilities
        """
        log_prob = self.log_prob(x)
        log_prob_norm = jax.scipy.special.logsumexp(
            log_prob, axis=Axis.components, keepdims=True
        )
        return log_prob_norm

    @jax.jit
    def score(self, x: jax.Array) -> jax.Array:
        """Compute the log likelihood of the data

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data
        """
        log_prob = self.score_samples(x)
        return jnp.mean(log_prob)

    @jax.jit
    def aic(self, x: jax.Array) -> jax.Array:
        """Compute the Akaike Information Criterion

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        aic : jax.array
            Akaike Information Criterion
        """
        return -2 * self.score(x) * x.shape[Axis.batch] + 2 * self.n_parameters  # type: ignore [no-any-return]

    @jax.jit
    def bic(self, x: jax.Array) -> jax.Array:
        """Compute the Bayesian Information Criterion

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        bic : jax.array
            Bayesian Information Criterion
        """
        return -2 * self.score(x) * x.shape[Axis.batch] + self.n_parameters * jnp.log(  # type: ignore [no-any-return]
            x.shape[Axis.batch]
        )

    @partial(jax.jit, static_argnames=["n_samples"])
    def sample(self, key: jax.Array, n_samples: int):
        """Sample from the model

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key
        n_samples : int
            Number of samples

        Returns
        -------
        samples : jax.array
            Samples
        """
        key, subkey = jax.random.split(key)

        selected = jax.random.choice(
            key,
            jnp.arange(self.n_components),
            p=self.weights.flatten(),
            shape=(n_samples,),
        )

        # TODO: this blows up the memory, as the arrays are copied, however
        # there is no simple way to handle the varying numbers of samples per component
        # Jax does not support ragged arrays and the size parameter in random methods has
        # to be static. One possibility would be to pad the arrays to the maximum number
        # of samples per component, however this might be inefficient as well.
        means = jnp.take(self.means, selected, axis=Axis.components)
        covar = jnp.take(self.covariances.values_dense, selected, axis=Axis.components)

        samples = jax.random.multivariate_normal(
            subkey,
            jnp.squeeze(means, axis=(Axis.batch, Axis.features_covar)),
            jnp.squeeze(covar, axis=Axis.batch),
            shape=(n_samples,),
        )

        return samples, selected

    @classmethod
    def from_fastinit(
        cls,
        x,
        n_components: int,
        reg_covar: float = 1e-6,
        covariance_type: CovarianceType = CovarianceType.full,
        **kwargs: dict,
    ) -> GaussianMixtureModelJax:
        kwargs['max_iter'] = 1
        kwargs['n_init'] = 1
        kwargs['tol'] = 1e-2
        kwargs['init'] = 'random'
        return cls.from_k_means(x, n_components, reg_covar, covariance_type, **kwargs)


def check_model_fitted(
    instance: GaussianMixtureSKLearn,
) -> GaussianMixtureModelJax:
    """Check if the model is fitted"""
    if instance._gmm is None:
        message = "Model not initialized. Call `fit` first."
        raise ValueError(message)

    return instance._gmm


INIT_METHODS = {
    "kmeans": GaussianMixtureModelJax.from_k_means,
    "fastinit": GaussianMixtureModelJax.from_fastinit,
}


@dataclass
class GaussianMixtureSKLearn:
    """Scikit learn compatibile API for Gaussian Mixture Model

    See docs at https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """

    n_components: int
    covariance_type: str = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: str = "kmeans"
    weights_init = None
    means_init = None
    precisions_init = None
    random_state: np.random.RandomState | None = None
    warm_start: bool = False
    _gmm: GaussianMixtureModelJax | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        from sklearn.utils import check_random_state  # type: ignore [import-untyped]
        self.random_state = check_random_state(self.random_state)

    @property
    def weights_(self):
        """Weights of each component"""
        return check_model_fitted(self).weights_numpy

    @property
    def means_(self):
        """Means of each component"""
        return check_model_fitted(self).means_numpy

    @property
    def precisions_cholesky_(self):
        """Precision matrices of each component"""
        return check_model_fitted(self).covariances.precisions_cholesky_numpy

    @property
    def covariances_(self):
        """Covariances of each component"""
        return check_model_fitted(self).covariances.values_numpy

    def _initialize_gmm(self, x) -> None:
        init_from_data = (
            self.weights_init is None
            or self.means_init is None
            or self.precisions_init is None
        )

        if init_from_data:
            kwargs = {
                "x": x,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
            }
            self._gmm = INIT_METHODS[self.init_params](**kwargs)  # type: ignore [arg-type]
        else:
            covar = COVARIANCE[CovarianceType(self.covariance_type)]

            self._gmm = GaussianMixtureModelJax.from_squeezed(
                means=self.means_init,  # type: ignore [arg-type]
                covariances=covar.from_precisions(
                    self.precisions_init.astype(np.float32)  # type: ignore [union-attr]
                ).values_numpy,
                weights=self.weights_init,  # type: ignore [arg-type]
                covariance_type=self.covariance_type,
            )

    def fit(self, X) -> GaussianMixtureSKLearn:
        """Fit the model"""
        from gmmx.fit import EMFitter

        do_init = not (self.warm_start and hasattr(self, "converged_"))

        if do_init:
            self._initialize_gmm(x=X)

        fitter = EMFitter(
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
        )
        result = fitter.fit(X, self._gmm)
        self._gmm = result.gmm
        self.converged_ = result.converged
        return self

    def predict(self, X):
        """Predict the component index for each sample"""
        return np.squeeze(check_model_fitted(self).predict(X), axis=Axis.components)  # type: ignore [no-any-return]

    def fit_predict(self):
        """Fit the model and predict the component index for each sample"""
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict the probability of each sample belonging to each component"""
        return np.squeeze(  # type: ignore [no-any-return]
            check_model_fitted(self).predict_proba(X),
            axis=(Axis.features, Axis.features_covar),
        )

    def sample(self, n_samples: int):
        """Sample from the model"""
        key = jax.random.key(self.random_state.randint(2**32 - 1))  # type: ignore [union-attr]
        samples, labels = check_model_fitted(self).sample(key=key, n_samples=n_samples)
        return np.asarray(samples), np.asarray(labels)

    def score(self, X):
        """Compute the log likelihood of the data"""
        return np.asarray(check_model_fitted(self).score(X))

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample"""
        return np.squeeze(  # type: ignore [no-any-return]
            (check_model_fitted(self).score_samples(X)),
            axis=(Axis.components, Axis.features, Axis.features_covar),
        )

    def bic(self, X):
        """Compute the Bayesian Information Criterion"""
        return np.asarray(check_model_fitted(self).bic(X))

    def aic(self, X):
        """Compute the Akaike Information Criterion"""
        return np.asarray(check_model_fitted(self).aic(X))
