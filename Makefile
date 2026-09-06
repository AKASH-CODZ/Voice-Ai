.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv, install backend + frontend deps
	python3 -m venv .venv
	./.venv/bin/pip install -U pip wheel
	./.venv/bin/pip install -r backend/requirements.txt
	cd frontend && npm install

models: ## Download Silero VAD + Kokoro TTS weights
	./.venv/bin/python tools/download_models.py

hw: ## Print the hardware diagnostic the router uses
	./.venv/bin/python tools/check_hardware.py

backend: ## Run FastAPI with reload
	./.venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000

frontend: ## Run the Next.js dev server
	cd frontend && npm run dev

mcp: ## Run the EchoSync MCP server over stdio
	./.venv/bin/python -m echosync_mcp.server

bench: ## Measure per-stage latency for both engines
	./.venv/bin/python tools/bench_latency.py

modes: ## Judge casual / teaching / observation against the live Ollama model
	./.venv/bin/python tools/check_modes.py

test: ## Run the test suite
	./.venv/bin/pytest tests -v

lint: ## Ruff + tsc
	./.venv/bin/ruff check backend mcp tools tests
	cd frontend && npx tsc --noEmit

up-gpu: ## Docker: local GPU profile
	docker compose -f docker/docker-compose.yml --profile gpu up --build

up-cloud: ## Docker: cloud/demo profile
	docker compose -f docker/docker-compose.yml --profile cloud up --build

down: ## Stop all containers
	docker compose -f docker/docker-compose.yml down

.PHONY: help setup models hw backend frontend mcp bench modes test lint up-gpu up-cloud down
