.PHONY: help setup install install-dev lint format typecheck test test-cov clean build train eval infer

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Set up development environment
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install
	python -m spacy download xx_ent_wiki_sm || echo "spaCy model download failed, will fallback gracefully"

install:  ## Install package
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

lint:  ## Run linting checks
	ruff check src tests
	black --check src tests
	mypy src

format:  ## Format code
	ruff check --fix src tests
	black src tests

typecheck:  ## Run type checking
	mypy src

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

# Data preparation
prepare-xlsum:  ## Prepare XL-Sum Hindi dataset
	ehi-prepare --dataset xlsum --lang hi --split train
	ehi-prepare --dataset xlsum --lang hi --split val
	ehi-prepare --dataset xlsum --lang hi --split test

prepare-ilsum:  ## Prepare ILSUM Hindi dataset  
	ehi-prepare --dataset ilsum --lang hi --split train
	ehi-prepare --dataset ilsum --lang hi --split val
	ehi-prepare --dataset ilsum --lang hi --split test

# Training
train:  ## Train EHI model with default config
	ehi-train --config configs/ehi_rl_mt5_small.yaml

train-resume:  ## Resume training from checkpoint
	ehi-train --config configs/ehi_rl_mt5_small.yaml --resume

# Evaluation
eval:  ## Evaluate model
	ehi-evaluate --config configs/eval.yaml --checkpoint outputs/best_model

eval-plot:  ## Evaluate and generate CDF plots
	ehi-evaluate --config configs/eval.yaml --checkpoint outputs/best_model --plot-cdfs

# Inference
infer:  ## Run inference on sample texts
	echo "यह एक परीक्षण वाक्य है।" | ehi-infer --checkpoint outputs/best_model

demo:  ## Run end-to-end demo
	jupyter notebook notebooks/demo_e2e.ipynb

# Development helpers
check-deps:  ## Check if all dependencies are properly installed
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import stanza; print('Stanza: OK')" || echo "Stanza not available"
	python -c "import spacy; print('spaCy: OK')" || echo "spaCy not available"

download-models:  ## Download required NLP models
	python -c "import stanza; stanza.download('hi')" || echo "Stanza Hindi download failed"
	python -m spacy download xx_ent_wiki_sm || echo "spaCy model download failed"

# CI/CD helpers
ci-setup:  ## Setup for CI environment
	pip install -e ".[dev]"
	python -c "import stanza; stanza.download('hi')" || echo "Stanza download failed in CI"

ci-test:  ## Run CI tests
	pytest tests/ -v --cov=src --cov-report=xml