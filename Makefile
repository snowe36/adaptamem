.PHONY: setup sim lint test gpu gpu-campaign gpu-runpod reproduce clean

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

gpu-campaign:
	python scripts/gpu_campaign.py

gpu-runpod:
	@echo "GPU oracle only. Ship assembled.pdb + system.xml first."
	@echo "Container command: bash scripts/runpod_boot.sh"
	@echo "Set ADAPTAMEM_WORKDIR and ORACLE_NS. No fetch/assemble/scout on the pod."
	@echo "Watchdog: scripts/runpod_watchdog.py --kill."

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
