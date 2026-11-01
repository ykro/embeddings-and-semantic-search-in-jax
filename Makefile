PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

experiments:
	$(PYTHON) experiments/01_tokens_to_embedding_table.py
	$(PYTHON) experiments/02_cosine_similarity_and_vector_space.py
	$(PYTHON) experiments/03_training_contrastive_embeddings.py
	$(PYTHON) experiments/04_jit_vector_search_engine.py
	$(PYTHON) experiments/05_zero_shot_and_linear_probe_classifier.py

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
