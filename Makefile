.DEFAULT_GOAL := help

SRC_DIR := carl
UV_RUN := uv run
UV_LINT_RUN := uv run --group lint
PYTHONPATH_PREFIX := PYTHONPATH=.

# Support both old lowercase vars and explicit uppercase vars.
RUN_DIR = $(strip $(or $(CONFIG_DIR),$(dir)))
RUN_NAME = $(strip $(or $(CONFIG_NAME),$(name)))
RUN_CONFIG_FILE = $(RUN_DIR)/$(RUN_NAME).yaml

.PHONY: help \
	check lint typecheck format ruff pyright mypy test \
	run_local_cpu run_local_gpu run_local_solve run_local_solve_gpu \
	validate_run_config

help: ## Show available targets and common examples
	@printf "CaRL Make Targets\n\n"
	@printf "Common usage:\n"
	@printf "  make run_local_gpu dir=configs/solve/sokoban name=sokoban_ada_solve\n"
	@printf "  make run_local_cpu CONFIG_DIR=configs/solve/sokoban CONFIG_NAME=sokoban_ada_solve\n"
	@printf "  make lint\n"
	@printf "  make format\n\n"
	@printf "Required vars for local run targets: dir + name (or CONFIG_DIR + CONFIG_NAME)\n"
	@printf "  - dir / CONFIG_DIR: directory containing Hydra config files\n"
	@printf "  - name / CONFIG_NAME: config basename without .yaml\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

validate_run_config:
	@if [ -z "$(RUN_DIR)" ] || [ -z "$(RUN_NAME)" ]; then \
		echo "Error: missing required variables for local run."; \
		echo "Required: dir=<config-dir> name=<config-name>"; \
		echo "Also supported: CONFIG_DIR=<config-dir> CONFIG_NAME=<config-name>"; \
		echo "Example: make run_local_gpu dir=configs/solve/sokoban name=sokoban_ada_solve"; \
		exit 1; \
	fi
	@if [ ! -d "$(RUN_DIR)" ]; then \
		echo "Error: config directory not found: $(RUN_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(RUN_CONFIG_FILE)" ]; then \
		echo "Error: config file not found: $(RUN_CONFIG_FILE)"; \
		echo "Hint: pass the config basename without .yaml in name/CONFIG_NAME."; \
		exit 1; \
	fi

run_local_cpu: validate_run_config ## Run a local config on CPU (requires dir=... name=...)
	$(PYTHONPATH_PREFIX) HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" $(UV_RUN) python3 -m carl.run --config-dir=$(RUN_DIR) --config-name $(RUN_NAME)

run_local_gpu: validate_run_config ## Run a local config on GPU (requires dir=... name=...)
	$(PYTHONPATH_PREFIX) HYDRA_FULL_ERROR=1 $(UV_RUN) python3 -m carl.run --config-dir=$(RUN_DIR) --config-name $(RUN_NAME)

run_local_solve: run_local_cpu ## Backward-compatible alias for run_local_cpu

run_local_solve_gpu: run_local_gpu ## Backward-compatible alias for run_local_gpu

test: ## Run tests
	$(UV_RUN) pytest tests

check: lint ## Alias for all static checks

lint: ruff pyright mypy ## Run linting and type checks

typecheck: pyright mypy ## Run pyright + mypy

ruff: ## Run Ruff checks on source tree only
	$(UV_LINT_RUN) ruff check $(SRC_DIR)

pyright: ## Run Pyright on source tree only
	$(PYTHONPATH_PREFIX) $(UV_RUN) pyright $(SRC_DIR)

mypy: ## Run Mypy on source tree only
	$(PYTHONPATH_PREFIX) $(UV_LINT_RUN) mypy $(SRC_DIR)

format: ## Format source tree only (Ruff formatter)
	$(UV_LINT_RUN) ruff format $(SRC_DIR)
