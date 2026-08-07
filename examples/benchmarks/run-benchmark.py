import getpass
import importlib
import json
import logging
import platform
import sys
import timeit
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax.lib import xla_bridge
from scipy import stats

from gmmx import EMFitter, GaussianMixtureModelJax

jax.config.update("jax_enable_x64", True)


PATH = Path(__file__).parent
PATH_RESULTS = PATH / "results"
RANDOM_STATE = np.random.RandomState(81737)
N_AVERAGE = 10
DPI = 180
KEY = jax.random.PRNGKey(81737)

MAX_ITER = 20
TOL = 0

PATH_TEMPLATE = "{user}-{machine}-{system}-{cpu}-{device-platform}"


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


Array = list[float]


def get_provenance():
    """Compute provenance info about software and data used."""
    env = {
        "user": getpass.getuser(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": platform.processor(),
        "device-platform": xla_bridge.get_backend().platform,
    }

    software = {
        "python-executable": sys.executable,
        "python-version": platform.python_version(),
        "jax-version": str(importlib.import_module("jax").__version__),
        "numpy-version": str(importlib.import_module("numpy").__version__),
        "sklearn-version": str(importlib.import_module("sklearn").__version__),
    }

    return {
        "env": env,
        "software": software,
    }


def gpu_is_available():
    """Check if a GPU is available"""
    try:
        jax.devices("gpu")
    except RuntimeError:
        return False

    return True


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    n_samples: Array
    n_components: Array
    n_features: Array
    time_sklearn: Array
    time_jax: Array
    time_jax_gpu: Optional[Array] = None
    provenance: dict = field(default_factory=get_provenance)

    def write_json(self, path):
        """Write the benchmark result to a JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def read(cls, path):
        """Read the benchmark result from a JSON file"""
        with Path(path).open("r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class BenchmarkSpec:
    """Benchmark specification"""

    filename_result: str
    n_components_grid: Array
    n_samples_grid: Array
    n_features_grid: Array
    x_axis: str
    title: str
    func_sklearn: Optional[callable] = None
    func_jax: Optional[callable] = None

    @property
    def path(self):
        """Absolute path to the benchmark result"""
        return (
            PATH_RESULTS
            / PATH_TEMPLATE.format(**get_provenance()["env"])
            / self.filename_result
        )


def predict_sklearn(gmm, x):
    """Predict the responsibilities"""
    return partial(gmm.predict, X=x)


def predict_jax(gmm, x):
    """Measure the time to predict the responsibilities"""

    def func():
        return gmm.predict(x=x).block_until_ready()

    return func


def fit_sklearn(gmm, x):
    """Measure the time to fit the model"""

    def func():
        result = gmm.fit(x)
        return result

    return func


def fit_jax(gmm, x):
    """Measure the time to fit the model"""

    def func():
        fitter = EMFitter(tol=TOL, max_iter=MAX_ITER)
        result = fitter.fit(x=x, gmm=gmm)
        return result

    return func


SPECS_PREDICT = [
    BenchmarkSpec(
        filename_result="time-vs-n-components-predict.json",
        n_components_grid=(2 ** np.arange(1, 7)).tolist(),
        n_samples_grid=[100_000],
        n_features_grid=[64],
        x_axis="n_components",
        title="Time vs Number of components",
        func_sklearn=predict_sklearn,
        func_jax=predict_jax,
    ),
    BenchmarkSpec(
        filename_result="time-vs-n-features-predict.json",
        n_components_grid=[128],
        n_samples_grid=[100_000],
        n_features_grid=(2 ** np.arange(1, 7)).tolist(),
        x_axis="n_features",
        title="Time vs Number of features",
        func_sklearn=predict_sklearn,
        func_jax=predict_jax,
    ),
    BenchmarkSpec(
        filename_result="time-vs-n-samples-predict.json",
        n_components_grid=[128],
        n_samples_grid=(2 ** np.arange(5, 18)).tolist(),
        n_features_grid=[64],
        x_axis="n_samples",
        title="Time vs Number of samples",
        func_sklearn=predict_sklearn,
        func_jax=predict_jax,
    ),
]

SPECS_FIT = [
    BenchmarkSpec(
        filename_result="time-vs-n-components-fit.json",
        n_components_grid=(2 ** np.arange(1, 6)).tolist(),
        n_samples_grid=[65536],
        n_features_grid=[32],
        x_axis="n_components",
        title="Time vs Number of components",
        func_sklearn=fit_sklearn,
        func_jax=fit_jax,
    ),
    BenchmarkSpec(
        filename_result="time-vs-n-features-fit.json",
        n_components_grid=[64],
        n_samples_grid=[65536],
        n_features_grid=(2 ** np.arange(1, 6)).tolist(),
        x_axis="n_features",
        title="Time vs Number of features",
        func_sklearn=fit_sklearn,
        func_jax=fit_jax,
    ),
    BenchmarkSpec(
        filename_result="time-vs-n-samples-fit.json",
        n_components_grid=[64],
        n_samples_grid=(2 ** np.arange(8, 17)).tolist(),
        n_features_grid=[32],
        x_axis="n_samples",
        title="Time vs Number of samples",
        func_sklearn=fit_sklearn,
        func_jax=fit_jax,
    ),
]


def create_random_gmm(n_components, n_features, random_state=RANDOM_STATE, device=None):
    """Create a random Gaussian mixture model"""
    means = random_state.uniform(-10, 10, (n_components, n_features))

    # the whishart distribution creates a positive semi-definite matrix
    covariances = stats.wishart.rvs(
        df=n_features,
        scale=np.eye(n_features) / n_features,
        size=n_components,
        random_state=random_state,
    )

    weights = random_state.uniform(0, 1, n_components)
    weights /= weights.sum()

    return GaussianMixtureModelJax.from_squeezed(
        means=jax.device_put(means, device=device),
        covariances=jax.device_put(covariances, device=device),
        weights=jax.device_put(weights, device=device),
    )


def create_random_data(n_samples, n_features, random_state=RANDOM_STATE, device=None):
    """Create random data"""
    return random_state.uniform(-10, 10, (n_samples, n_features))


def get_meta_str(result, x_axis):
    """Get the metadata for the benchmark result"""
    if x_axis == "n_samples":
        meta = {"n_components": result.n_components, "n_features": result.n_features}

    elif x_axis == "n_components":
        meta = {"n_samples": result.n_samples, "n_features": result.n_features}

    elif x_axis == "n_features":
        meta = {"n_samples": result.n_samples, "n_components": result.n_components}
    else:
        message = f"Invalid x_axis: {x_axis}"
        raise ValueError(message)

    return ", ".join(f"{k}={v[0]}" for k, v in meta.items())


def plot_result(result, x_axis, filename, title=""):
    """Plot the benchmark result"""
    log.info(f"Plotting {filename}")
    _, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = getattr(result, x_axis)
    meta = get_meta_str(result, x_axis)

    color = "#F1C44D"
    ax.plot(x, result.time_sklearn, label=f"sklearn ({meta})", color=color)
    ax.scatter(x, result.time_sklearn, color=color)

    color = "#405087"
    ax.plot(x, result.time_jax, label=f"jax ({meta})", color=color, zorder=3)
    ax.scatter(x, result.time_jax, color=color, zorder=3)

    if result.time_jax_gpu:
        color = "#E58336"
        ax.plot(
            x, result.time_jax_gpu, label=f"jax-gpu ({meta})", color=color, zorder=5
        )
        ax.scatter(x, result.time_jax_gpu, color=color, zorder=5)

    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Time (s)")
    ax.semilogx()
    ax.semilogy()
    ax.legend()

    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=DPI)


def measure_time(func):
    """Measure the time to run a function"""
    timer = timeit.Timer(func)
    return timer.timeit(N_AVERAGE)


def measure_time_sklearn_vs_jax(
    n_components_grid,
    n_samples_grid,
    n_features_grid,
    init_func_sklearn=predict_sklearn,
    init_func_jax=predict_jax,
):
    """Measure the time to predict the responsibilities for sklearn and jax"""
    time_sklearn, time_jax, time_jax_gpu = [], [], []

    for n_component, n_samples, n_features in product(
        n_components_grid, n_samples_grid, n_features_grid
    ):
        log.info(
            f"Running n_components={n_component}, n_samples={n_samples}, n_features={n_features}"
        )
        gmm = create_random_gmm(n_component, n_features, device=jax.devices("cpu")[0])
        x, _ = gmm.to_sklearn(random_state=RANDOM_STATE).sample(n_samples)

        func_sklearn = init_func_sklearn(gmm.to_sklearn(tol=TOL, max_iter=MAX_ITER), x)
        func_jax = init_func_jax(gmm, jax.device_put(x, device=jax.devices("cpu")[0]))

        time_sklearn.append(measure_time(func_sklearn))
        time_jax.append(measure_time(func_jax))

        if gpu_is_available():
            gmm_gpu = create_random_gmm(
                n_component, n_features, device=jax.devices("gpu")[0]
            )
            x_gpu = jax.device_put(x, device=jax.devices("gpu")[0])
            func_jax = init_func_jax(gmm_gpu, x_gpu)
            time_jax_gpu.append(measure_time(func_jax))

    return BenchmarkResult(
        n_samples=n_samples_grid,
        n_components=n_components_grid,
        n_features=n_features_grid,
        time_sklearn=time_sklearn,
        time_jax=time_jax,
        time_jax_gpu=time_jax_gpu or None,
    )


def run_benchmark_from_spec(spec):
    """Run a benchmark from a specification"""
    if not spec.path.exists():
        result = measure_time_sklearn_vs_jax(
            spec.n_components_grid,
            spec.n_samples_grid,
            spec.n_features_grid,
            init_func_sklearn=spec.func_sklearn,
            init_func_jax=spec.func_jax,
        )
        result.write_json(spec.path)

    result = BenchmarkResult.read(spec.path)
    plot_result(
        result,
        x_axis=spec.x_axis,
        filename=spec.path.with_suffix(".png"),
        title=spec.title,
    )


if __name__ == "__main__":
    for spec in SPECS_PREDICT:
        run_benchmark_from_spec(spec)

    for spec in SPECS_FIT:
        run_benchmark_from_spec(spec)
