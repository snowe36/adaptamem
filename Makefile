.PHONY: setup lint test reproduce clean

setup:
	uv venv --python 3.11
	uv sync --extra dev

lint:
	uv run ruff check .

test:
	uv run pytest -q

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
