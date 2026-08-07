.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

# Each version gets its own environment, so that the default .venv is left alone
PYTHON_VERSIONS ?= 3.9 3.10 3.11 3.12 3.13

.PHONY: test-all
test-all: ## Test the code against all supported Python versions
	@for version in $(PYTHON_VERSIONS); do \
		echo "🚀 Testing code: Running pytest and mypy on Python $$version"; \
		UV_PROJECT_ENVIRONMENT=.venvs/$$version uv run --python $$version \
			python -m pytest || exit 1; \
		UV_PROJECT_ENVIRONMENT=.venvs/$$version uv run --python $$version \
			mypy || exit 1; \
	done

# Older dependencies do not have wheels for the newest Python versions, so both
# targets below pin an interpreter instead of using whatever .venv happens to be.
COMPAT_PYTHON_VERSION ?= 3.11

# Note that not every jax version supports every Python version, e.g. jax 0.11
# requires Python 3.12. The versions here all work on the interpreter above.
JAX_VERSIONS ?= 0.4.30 0.6.2 0.9.2

.PHONY: test-jax
test-jax: ## Test the code against multiple jax versions
	@for version in $(JAX_VERSIONS); do \
		echo "🚀 Testing code: Running pytest against jax $$version"; \
		UV_PROJECT_ENVIRONMENT=.venvs/$(COMPAT_PYTHON_VERSION) \
			uv run --python $(COMPAT_PYTHON_VERSION) --with jax==$$version \
			python -m pytest || exit 1; \
	done

.PHONY: test-lowest
test-lowest: ## Test the code against the lowest declared dependency versions
	@echo "🚀 Testing code: Running pytest against the lowest direct dependencies"
	@UV_PROJECT_ENVIRONMENT=.venvs/lowest uv run --python $(COMPAT_PYTHON_VERSION) \
		--resolution lowest-direct --isolated python -m pytest

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
