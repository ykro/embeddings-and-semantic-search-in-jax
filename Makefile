PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
OUT := benchmarks/experiment_outputs

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

# Each run is also saved to benchmarks/experiment_outputs/ so every number is traceable.
experiments:
	mkdir -p $(OUT)
	$(PYTHON) experiments/01_tokens_to_embedding_table.py --show-ir | tee $(OUT)/01_tokens_to_embedding_table.txt
	$(PYTHON) experiments/02_cosine_similarity_and_vector_space.py | tee $(OUT)/02_cosine_similarity_and_vector_space.txt
	$(PYTHON) experiments/03_training_contrastive_embeddings.py | tee $(OUT)/03_training_contrastive_embeddings.txt
	$(PYTHON) experiments/04_jit_vector_search_engine.py --show-ir | tee $(OUT)/04_jit_vector_search_engine.txt
	$(PYTHON) experiments/05_zero_shot_and_linear_probe_classifier.py | tee $(OUT)/05_zero_shot_and_linear_probe_classifier.txt

verify:
	$(PYTHON) verify_embeddings.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

gcp-gpu:
	bash gcp/run_on_demand_embeddings.sh gpu

gcp-tpu:
	bash gcp/run_on_demand_embeddings.sh tpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
