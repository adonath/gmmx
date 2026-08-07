import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gmmx import EMFitter, GaussianMixtureModelJax, GaussianMixtureSKLearn
from gmmx.gmm import SphericalCovariances

TEST_COVARIANCES = {
    "full": np.array([
        [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
        [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
    ]),
    "diag": np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    "tied": np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]),
    "spherical": np.array([1.0, 2.0]),
}

TEST_PRECISIONS = {
    "full": np.linalg.inv(TEST_COVARIANCES["full"]),
    "diag": 1 / TEST_COVARIANCES["diag"],
    "tied": np.linalg.inv(TEST_COVARIANCES["tied"]),
    "spherical": 1 / TEST_COVARIANCES["spherical"],
}

MEANS = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
WEIGHTS = np.array([0.2, 0.8])

# Note: the dtype matters, as sklearn allocates intermediate arrays using the
# dtype of the input data, which silently truncates the result for integers.
X = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 4.0, 2.0],
    [1.0, 0.0, 6.0],
    [4.0, 2.0, 4.0],
    [4.0, 4.0, 4.0],
    [4.0, 0.0, 2.0],
])


@pytest.fixture(params=["full", "diag", "tied", "spherical"])
def gmm_jax(request):
    covariances = TEST_COVARIANCES[request.param]
    return GaussianMixtureModelJax.from_squeezed(
        means=MEANS,
        covariances=covariances,
        weights=WEIGHTS,
        covariance_type=request.param,
    )


def test_simple(gmm_jax):
    assert gmm_jax.n_features == 3
    assert gmm_jax.n_components == 2

    expected = {"full": 19, "diag": 13, "tied": 13, "spherical": 9}
    assert gmm_jax.n_parameters == expected[gmm_jax.covariances.type.value]

    assert (
        gmm_jax.covariances.values_numpy.shape
        == TEST_COVARIANCES[gmm_jax.covariances.type.value].shape
    )


@pytest.mark.parametrize(
    ("covariance_type", "n_parameters", "n_covariances"),
    [("full", 17, 3), ("diag", 14, 3), ("tied", 11, 1), ("spherical", 11, 3)],
)
def test_create(covariance_type, n_parameters, n_covariances):
    gmm = GaussianMixtureModelJax.create(
        n_components=3, n_features=2, covariance_type=covariance_type
    )
    assert gmm.n_features == 2
    assert gmm.n_components == 3
    assert gmm.n_parameters == n_parameters
    assert_allclose(
        gmm.covariances.values_dense, np.stack([np.eye(2)] * n_covariances)[None]
    )


def test_init_incorrect():
    with pytest.raises(ValueError):
        GaussianMixtureModelJax(
            means=jnp.zeros((2, 3)),
            covariances=jnp.zeros((2, 3, 3)),
            weights=jnp.zeros((2,)),
        )

    with pytest.raises(ValueError):
        GaussianMixtureModelJax(
            means=jnp.zeros((1, 2, 3, 1)),
            covariances=jnp.zeros((1, 2, 3, 3)),
            weights=jnp.zeros((1, 1, 4, 1)),
        )


def test_spherical_requires_n_features():
    # the squeezed shape of the spherical covariances does not encode the
    # number of features, so it has to be passed explicitly
    with pytest.raises(ValueError, match="n_features is required"):
        SphericalCovariances.from_squeezed(TEST_COVARIANCES["spherical"])

    covariances = SphericalCovariances.from_squeezed(
        TEST_COVARIANCES["spherical"], n_features=3
    )
    assert covariances.n_features == 3
    assert covariances.n_components == 2
    assert_allclose(covariances.values_numpy, TEST_COVARIANCES["spherical"])


def test_against_sklearn(gmm_jax):
    x = X

    gmm = gmm_jax.to_sklearn()
    result_ref = gmm._estimate_weighted_log_prob(X=x)
    result = gmm_jax.log_prob(x=jnp.asarray(x))[:, :, 0, 0]

    assert_allclose(np.asarray(result), result_ref, rtol=1e-6)

    assert gmm_jax.n_parameters == gmm._n_parameters()


@pytest.mark.parametrize(
    "method", ["aic", "bic", "predict", "predict_proba", "score", "score_samples"]
)
def test_against_sklearn_all(gmm_jax, method):
    gmm = gmm_jax.to_sklearn()
    x = X
    result_sklearn = getattr(gmm, method)(x)
    result_jax = getattr(gmm_jax, method)(jnp.asarray(x))
    assert_allclose(np.squeeze(result_jax), result_sklearn, rtol=1e-5)


def test_sample(gmm_jax):
    key = jax.random.PRNGKey(0)

    samples, selected = gmm_jax.sample(key, 2)
    assert samples.shape == (2, 3)
    assert selected.shape == (2,)

    # The individual sample values depend on the random number stream of the
    # installed JAX version, so we compare the empirical distribution instead.
    samples, selected = gmm_jax.sample(key, 100_000)
    samples, selected = np.asarray(samples), np.asarray(selected)

    mean_expected = np.sum(WEIGHTS[:, np.newaxis] * MEANS, axis=0)
    diff = MEANS - mean_expected
    covariances = np.broadcast_to(
        np.asarray(gmm_jax.covariances.values_dense)[0],
        (len(WEIGHTS), gmm_jax.n_features, gmm_jax.n_features),
    )
    covar_expected = np.einsum("k,kij->ij", WEIGHTS, covariances) + np.einsum(
        "k,ki,kj->ij", WEIGHTS, diff, diff
    )

    assert_allclose(samples.mean(axis=0), mean_expected, atol=0.05)
    assert_allclose(np.cov(samples.T), covar_expected, atol=0.05)

    # the samples of each component follow the mean of that component
    fraction = np.bincount(selected, minlength=len(WEIGHTS)) / selected.size
    assert_allclose(fraction, WEIGHTS, atol=0.01)

    for idx, mean in enumerate(MEANS):
        assert_allclose(samples[selected == idx].mean(axis=0), mean, atol=0.05)


def test_predict(gmm_jax):
    result = gmm_jax.predict(x=jnp.asarray(X))

    assert result.shape == (6, 1)

    # for the spherical case the second, wider component dominates
    expected = {"full": 0, "diag": 0, "tied": 0, "spherical": 1}
    assert_allclose(result[0], expected[gmm_jax.covariances.type.value], rtol=1e-6)


def test_fit(gmm_jax):
    random_state = np.random.RandomState(827392)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    fitter = EMFitter(tol=1e-6)
    result = fitter.fit(x=x, gmm=gmm_jax)

    # The number of iterations is not deterministic across architectures
    # and depends on the precision the model is fitted with
    covar_str = gmm_jax.covariances.type.value
    expected = {"full": [4, 7], "diag": [6, 7], "tied": [4, 5], "spherical": [13, 14]}
    assert int(result.n_iter) in expected[covar_str]

    expected = {
        "full": -4.3686,
        "diag": -5.422534,
        "tied": -4.368919,
        "spherical": -5.381288,
    }
    assert_allclose(result.log_likelihood, expected[covar_str], rtol=2e-4)

    expected = dict.fromkeys(["full", "diag", "tied", "spherical"], 9.536743e-07)
    assert_allclose(result.log_likelihood_diff, expected[covar_str], atol=fitter.tol)

    assert_allclose(result.gmm.weights_numpy, [0.2, 0.8], rtol=0.06)


def test_move_to_device(gmm_jax):
    gmm_jax_moved = gmm_jax.to_device(jax.devices()[0])
    assert_allclose(gmm_jax.means_numpy, gmm_jax_moved.means_numpy)
    assert gmm_jax_moved.means.devices() == {jax.devices()[0]}


def test_io(gmm_jax, tmpdir):
    filename = tmpdir / "model.safetensors"

    gmm_jax.write(filename)

    new_model = GaussianMixtureModelJax.read(filename)

    assert_allclose(gmm_jax.means_numpy, new_model.means_numpy)
    assert_allclose(gmm_jax.weights_numpy, new_model.weights_numpy)
    assert_allclose(
        gmm_jax.covariances.values_numpy, new_model.covariances.values_numpy
    )

    assert gmm_jax.covariances.type == new_model.covariances.type


def test_fit_against_sklearn(gmm_jax):
    # Fitting is hard to test, especillay we cannot guarantee the fit converges to the same solution
    # However the "global" likelihood (summed accross all components) for a given feature vector
    # should be similar for both implementations
    random_state = np.random.RandomState(82792)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    tol = 1e-6
    fitter = EMFitter(tol=tol)
    result_jax = fitter.fit(x=x, gmm=gmm_jax)

    gmm_sklearn = gmm_jax.to_sklearn(tol=tol, random_state=random_state)

    # This brings the sklearn model in the same state as the jax model
    gmm_sklearn.fit(x)

    covar_str = gmm_jax.covariances.type.value

    expected = {"full": 9, "diag": 9, "tied": 5, "spherical": 16}
    assert_allclose(gmm_sklearn.n_iter_, expected[covar_str])
    assert_allclose(gmm_sklearn.weights_, [0.2, 0.8], rtol=0.06)

    expected = {"full": [9], "diag": [8, 9, 11], "tied": [5, 6], "spherical": [15, 16]}
    assert result_jax.n_iter in expected[covar_str]
    assert_allclose(result_jax.gmm.weights_numpy, [0.2, 0.8], rtol=0.06)

    assert_allclose(gmm_sklearn.covariances_, TEST_COVARIANCES[covar_str], rtol=0.1)
    assert_allclose(
        result_jax.gmm.covariances.values_numpy, TEST_COVARIANCES[covar_str], rtol=0.1
    )

    log_likelihood_jax = result_jax.gmm.log_prob(x[:10]).sum(axis=1)[:, 0, 0]
    log_likelihood_sklearn = gmm_sklearn._estimate_weighted_log_prob(x[:10]).sum(axis=1)

    # note this is agreement in log-likehood, not likelihood!
    assert_allclose(log_likelihood_jax, log_likelihood_sklearn, rtol=1e-2)


def test_sklearn_api(gmm_jax):
    random_state = np.random.RandomState(829282)
    x = X

    covar_str = gmm_jax.covariances.type.value

    gmm = GaussianMixtureSKLearn(
        n_components=2,
        covariance_type=covar_str,
        tol=1e-6,
        random_state=random_state,
        weights_init=WEIGHTS,
        means_init=MEANS,
        precisions_init=TEST_PRECISIONS[covar_str],
        max_iter=0,  # we just want to test the API, so we don't want to fit
    )
    gmm.fit(x)

    assert not gmm.converged_

    assert_allclose(gmm.weights_, WEIGHTS, rtol=0.06)
    assert_allclose(gmm.covariances_, TEST_COVARIANCES[covar_str], rtol=0.1)
    assert_allclose(gmm.means_, MEANS, atol=0.05)

    value = gmm.score_samples(x[:2])
    expected = {
        "full": [-7.018339, -12.698977],
        "diag": [-8.909672, -11.101017],
        "tied": [-7.018339, -12.698976],
        "spherical": [-8.788465, -10.259644],
    }
    assert_allclose(value, expected[covar_str], rtol=1e-4)

    value = gmm.score(x[:2])
    expected = {
        "full": -9.858658,
        "diag": -10.005345,
        "tied": -9.858657,
        "spherical": -9.524054,
    }
    assert_allclose(value, expected[covar_str], rtol=1e-4)

    value = gmm.predict(x[:2])
    expected = {"full": [0, 0], "diag": [0, 0], "tied": [0, 0], "spherical": [1, 1]}
    assert_allclose(value, expected[covar_str], rtol=1e-4)

    value = gmm.predict_proba(x[:2])
    expected = {
        "full": [[0.99866, 0.00134], [0.931738, 0.068262]],
        "diag": [[0.981054, 0.018946], [0.720519, 0.279481]],
        "tied": [[0.99866, 0.00134], [0.931738, 0.068262]],
        "spherical": [[0.206431, 0.793568], [0.009986, 0.990015]],
    }
    assert_allclose(value, expected[covar_str], atol=1e-3)

    value = gmm.aic(x[:2])
    expected = {
        "full": 77.434632,
        "diag": 66.021378,
        "tied": 65.43463,
        "spherical": 56.096214,
    }
    assert_allclose(value, expected[covar_str], rtol=1e-4)

    value = gmm.bic(x[:2])
    expected = {
        "full": 52.604428,
        "diag": 49.032291,
        "tied": 48.44554,
        "spherical": 44.334538,
    }
    assert_allclose(value, expected[covar_str], rtol=1e-4)

    # same signature as `~sklearn.mixture.GaussianMixture.sample`
    samples, selected = gmm.sample(10)
    assert samples.shape == (10, 3)
    assert selected.shape == (10,)
    assert set(np.unique(selected)) <= {0, 1}
