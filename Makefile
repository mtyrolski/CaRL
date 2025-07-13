.PHONY: help
help:
	@echo "Usage: make run_local_solve CONFIG=YourName"

run_local_solve:
	HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" python3 -m carl.run --config-dir=${dir} --config-name ${name}

run_local_solve_gpu:
	HYDRA_FULL_ERROR=1 python3 -m carl.run --config-dir=${dir} --config-name ${name}

test:
	PYTHONPATH=. pytest .

lint:
	@echo
	ruff . --fix
	@echo
	mypy .

# pyright:
#	@echo
#	poetry run pyright carl --pythonpath . --stats
