.PHONY: setup sim lint test gpu reproduce clean

setup:
	uv venv --python 3.11
	uv sync --extra dev

sim:
	uv sync --extra dev --extra sim

lint:
	uv run ruff check .

test:
	uv run pytest -q

gpu:
	uv run --extra sim python scripts/gpu_job.py --out runs/gpu --steps 4000

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
