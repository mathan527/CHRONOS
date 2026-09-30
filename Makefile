# Thin wrapper over scripts/tasks.py, which also works where `make` doesn't exist (Windows):
#   python scripts/tasks.py setup|test|run|bench|demo|docker
PYTHON ?= python3
TASK = $(PYTHON) scripts/tasks.py

.PHONY: setup test run bench demo docker

setup:   ## create .venv and install CHRONOS + dev tools
	$(TASK) setup

test:    ## pytest -q + ruff check
	$(TASK) test

run:     ## API + web client on http://localhost:8000 (CHRONOS_LLM=ollama for real models)
	$(TASK) run

bench:   ## full benchmark -> bench/results.md, results.json, charts/
	$(TASK) bench

demo:    ## play the 4 scenarios, write timelines, then serve the UI
	$(TASK) demo

docker:  ## docker compose up --build (Ollama + model pull + CHRONOS)
	$(TASK) docker
