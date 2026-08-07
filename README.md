# GMMX: Gaussian Mixture Models in Jax

[![Release](https://img.shields.io/github/v/release/adonath/gmmx)](https://img.shields.io/github/v/release/adonath/gmmx)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/gmmx/main.yml?branch=main)](https://github.com/adonath/gmmx/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/gmmx/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/gmmx)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/gmmx)](https://img.shields.io/github/commit-activity/m/adonath/gmmx)
[![License](https://img.shields.io/github/license/adonath/gmmx)](https://img.shields.io/github/license/adonath/gmmx)
[![DOI](https://zenodo.org/badge/879790145.svg)](https://doi.org/10.5281/zenodo.14515326)

<p align="center">
<img width="50%" src="https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/gmmx-logo.png" alt="GMMX Logo"/>
</p>

A minimal implementation of Gaussian Mixture Models in Jax

- **Github repository**: <https://github.com/adonath/gmmx/>
- **Documentation** <https://adonath.github.io/gmmx/>

## Installation

`gmmx` can be installed via pip:

```bash
pip install gmmx
```

Or alternatively you can use `conda/mamba`:

```bash
conda install gmmx
```

## Usage

```python
import jax
from gmmx import GaussianMixtureModelJax, EMFitter

# Create a Gaussian Mixture Model with 16 components and 32 features
gmm = GaussianMixtureModelJax.create(n_components=16, n_features=32)

# Draw samples from the model, along with the component they were drawn from
n_samples = 10_000
x, selected = gmm.sample(jax.random.PRNGKey(0), n_samples)

# Fit the model to the data
em_fitter = EMFitter(tol=1e-3, max_iter=100)
gmm_fitted = em_fitter.fit(x=x, gmm=gmm)
```

Alternatively the model can be fitted by gradient descent, using the Adam optimizer from [optax](https://optax.readthedocs.io):

```python
from gmmx import AdamFitter

adam_fitter = AdamFitter(tol=1e-6, max_iter=1_000, learning_rate=1e-2)
gmm_fitted = adam_fitter.fit(x=x, gmm=gmm)
```

Both fitters share the same interface and return the same result object.
The `EMFitter` converges in far fewer iterations and should be the first choice.
The `AdamFitter` does not rely on the closed form solution of the maximization step, which makes it easier to extend, for example to a fit of only a subset of the parameters.

If you use the code in a scientific publication, please cite the Zenodo DOI from the badge above.

## Why Gaussian Mixture models?

What are Gaussian Mixture Models (GMM) useful for in the age of deep learning? GMMs might have come out of fashion for classification tasks, but they still
have a few properties that make them useful in certain scenarios:

- They are universal approximators, meaning that given enough components they can approximate any distribution.
- Their likelihood can be evaluated in closed form, which makes them useful for generative modeling.
- They are rather fast to train and evaluate.

I would strongly recommend to read [In Depth: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html) from the Python Data Science Handbook for a more in-depth introduction to GMMs and their
application as density estimators.

One of these applications in my research is the context of image reconstruction, where GMMs can be used to model the distribution and pixel correlations of local (patch based)
image features. This can be useful for tasks like image denoising or inpainting. One of these methods I have used them for is [Jolideco](https://github.com/jolideco/jolideco).
Speed up the training of O(10^6) patches was the main motivation for `gmmx`.

## Benchmarks

Here are some results from the benchmarks in the [examples/benchmarks](https://github.com/adonath/gmmx/tree/main/examples/benchmarks) folder comparing against Scikit-Learn. The benchmarks were run on an "Intel(R) Xeon(R) Gold 6338" CPU and a single "NVIDIA L40S" GPU.

### Prediction Time

| Time vs. Number of Components                                                                                                       | Time vs. Number of Samples                                                                                                    | Time vs. Number of Features                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| ![Time vs. Number of Components](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-components-predict.png) | ![Time vs. Number of Samples](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-samples-predict.png) | ![Time vs. Number of Features](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-features-predict.png) |

For prediction the speedup is around 5-6x for varying number of components and features and ~50x speedup on the GPU. For the number of samples the cross-over point is around O(10^3) samples.

### Training Time

| Time vs. Number of Components                                                                                                   | Time vs. Number of Samples                                                                                                | Time vs. Number of Features                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| ![Time vs. Number of Components](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-components-fit.png) | ![Time vs. Number of Samples](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-samples-fit.png) | ![Time vs. Number of Features](https://raw.githubusercontent.com/adonath/gmmx/main/docs/_static/time-vs-n-features-fit.png) |

For training the speedup is around ~5-6x on the same architecture and ~50x speedup on the GPU. In the bechmark I have forced both fitters to evaluate exactly the same number of iterations. However in general there is no guarantee that GMMX converges to the same solution as Scikit-Learn. But there are some tests in the `tests` folder that compare the results of the two implementations which shows good agreement.
