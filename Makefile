# Makefile pour Quantum Retro-Causal Engine

.PHONY: install test lint format run clean

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/

format:
	black src/ tests/

run:
	python -m src.quantum_engine.main

test:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache logs/* checkpoints/* backups/* 