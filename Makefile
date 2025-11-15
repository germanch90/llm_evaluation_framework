.PHONY: help setup test clean docker-build docker-up docker-down lint format

help:
	@echo "Available commands:"
	@echo "  setup          - Set up development environment"
	@echo "  test           - Run all tests"
	@echo "  test-eval      - Run evaluation tests only"
	@echo "  clean          - Clean up generated files"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start Docker containers"
	@echo "  docker-down    - Stop Docker containers"
	@echo "  lint           - Run code linters"
	@echo "  format         - Format code"
	@echo "  download-data  - Download KG-RAG dataset"

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest

test-eval:
	pytest tests/evaluation/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov/ .coverage
	rm -rf logs/*.log

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

lint:
	flake8 src/ tests/
	mypy src/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

download-data:
	python scripts/download_dataset.py
