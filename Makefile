.PHONY: help run_local_solve run_local_solve_gpu test lint format ruff pyright mypy
help:
	@echo "Usage: make run_local_solve CONFIG=YourName"

run_local_solve:
	PYTHONPATH=. HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" uv run python3 -m carl.run --config-dir=${dir} --config-name ${name}

run_local_solve_gpu:
	PYTHONPATH=. HYDRA_FULL_ERROR=1 uv run python3 -m carl.run --config-dir=${dir} --config-name ${name}

test:
	uv run pytest .

lint: ruff pyright mypy

ruff:
	uv run --group lint ruff check .

pyright:
	PYTHONPATH=. uv run pyright 

mypy:
	PYTHONPATH=. uv run --group lint mypy carl

format:
	uv run --group lint ruff format .

# pyright:
#	@echo
#	poetry run pyright carl --pythonpath . --stats
# __init__.py  algorithms  environment           memory    run.py  solver
# __pycache__  dataloader  inference_components  planners  slurm   utils
