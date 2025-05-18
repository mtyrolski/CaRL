.PHONY: help
help:
	@echo "Usage: make run_local_solve CONFIG=YourName"

run_local_solve:
	HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" python3 -m carl.run --config-dir=${dir} --config-name ${name}

test:
	pytest .

test_supervised:
	pytest ./tests/supervised

test_memory:
	pytest ./tests/memory

lint:
	@echo
	ruff .
	@echo
	blue --check --diff --color .
	@echo
	mypy .

format:
	ruff --exit-zero --fix .
	blue .
