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
	@echo "Build docker/gpu.Dockerfile (openmm[cuda12] baked in) and start a Secure RTX 4090."
	@echo "Container command: bash scripts/runpod_boot.sh"
	@echo "Do not set pip or GPU_JOB_B64 in pod env."
	@echo "After CAMPAIGN_DONE, copy runs/1afo/compare.json and delete the pod."
	@echo "Fallback if the image cannot be pushed: git clone --depth 1 of this SHA, then pip install -e '.[gpu]' once — never stale main."

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

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
