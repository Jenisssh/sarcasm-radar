.PHONY: help install install-dev data train test test-cov lint format type serve app docker-build docker-up clean

help:
	@echo "sarcasm-radar — common tasks"
	@echo "  make install        editable install (runtime only)"
	@echo "  make install-dev    editable install + dev tools + pre-commit hooks"
	@echo "  make data           fetch iSarcasm + SemEval datasets"
	@echo "  make train          train DistilBERT on the merged corpus"
	@echo "  make test           run pytest"
	@echo "  make test-cov       pytest with coverage"
	@echo "  make lint           ruff check"
	@echo "  make format         ruff format"
	@echo "  make type           mypy strict"
	@echo "  make serve          FastAPI on :8000"
	@echo "  make app            Streamlit on :8501"
	@echo "  make docker-build   build the docker image"
	@echo "  make docker-up      docker compose up"
	@echo "  make clean          remove caches and artifacts"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

data:
	python -m scripts.download_data

train:
	python -m scripts.train_transformer --config configs/distilbert.yaml

test:
	pytest

test-cov:
	pytest --cov=src/sarcasm_radar --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

format:
	ruff format src tests

type:
	mypy src

serve:
	uvicorn sarcasm_radar.api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t sarcasm-radar:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
