.PHONY: install test demo api ui docker-build up down logs

install:        ## local dev install
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest -q

demo:           ## run the full demo scenario locally (offline mock LLM)
	python -m forensics.cli demo

api:
	uvicorn forensics.api:app --reload --port 8000

ui:
	FORENSICS_API_URL=http://localhost:8000 streamlit run ui/app.py

docker-build:
	docker build -t ai-failure-forensics:latest .

up:             ## API on :8000, trace explorer on :8501
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f
