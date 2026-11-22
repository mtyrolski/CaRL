.DEFAULT_GOAL := help

SRC_DIR := carl
SCRIPTS_DIR := scripts
UV_RUN := uv run
UV_LINT_RUN := uv run --group lint
PYTHONPATH_PREFIX := PYTHONPATH=.

# Support both old lowercase vars and explicit uppercase vars.
RUN_DIR = $(strip $(or $(CONFIG_DIR),$(dir)))
RUN_NAME = $(strip $(or $(CONFIG_NAME),$(name)))
RUN_CONFIG_FILE = $(RUN_DIR)/$(RUN_NAME).yaml

.PHONY: help pylint-icl icl-check icl-eval icl-eval-demo \
	check lint typecheck format ruff pyright mypy test \
	run_local_cpu run_local_gpu run_local_solve run_local_solve_gpu \
	validate_run_config show_active_experiments

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

show_active_experiments: ## Show supervised local runs, progress and report paths
	@python3 -m scripts.show_active_experiments $(EXPERIMENT_ARGS)

test: ## Run tests
	$(UV_RUN) --extra icl pytest tests

check: lint ## Alias for all static checks

lint: ruff pyright mypy ## Run linting and type checks

typecheck: pyright mypy ## Run pyright + mypy

ruff: ## Run Ruff checks on source and ICL entry points
	$(UV_LINT_RUN) ruff check $(SRC_DIR) $(SCRIPTS_DIR)

pylint-icl: ## Check the ICL implementation and primary CLI with the repository Pylint rules
	$(UV_RUN) --extra icl pylint carl/icl scripts/run_icl_search.py scripts/run_budget_sweep.py scripts/estimate_icl_tokens.py scripts/evaluate_icl.py

icl-check: ruff pyright pylint-icl test ## Verify ICL style, types and the full regression suite

pyright: ## Run Pyright on source, scripts and ICL tests
	$(PYTHONPATH_PREFIX) $(UV_RUN) --extra icl --extra lora pyright $(SRC_DIR) scripts
	$(PYTHONPATH_PREFIX) $(UV_RUN) --extra icl --extra lora pyright --project pyrightconfig-icl-tests.json

mypy: ## Run Mypy on source tree only
	$(PYTHONPATH_PREFIX) $(UV_LINT_RUN) mypy $(SRC_DIR)

format: ## Format source tree (Black; pinned Ruff predates its formatter)
	$(UV_LINT_RUN) black $(SRC_DIR)

icl-prepare: ## Prepare disjoint examples and verified boards through 32x32x10
	$(UV_RUN) --extra icl python -m scripts.run_icl_search prepare

icl-preview: ## Inspect prompts without sending requests
	$(UV_RUN) --extra icl python -m scripts.run_icl_search preview

icl-smoke: ## Verify all-Luna plumbing offline (not model performance)
	$(UV_RUN) --extra icl python -m scripts.run_icl_search calibrate --mock
	$(UV_RUN) --extra icl python -m scripts.run_icl_search matrix --mock --matrix configs/icl/experiments/matrix_smoke.json

icl-calibrate: ## Calibrate the frozen Luna policy using real API calls
	$(UV_RUN) --extra icl python -m scripts.run_icl_search calibrate --execute

icl-pilot: ## Run the paired 12x12 and 32x32 pilot via API after calibration
	$(UV_RUN) --extra icl python -m scripts.run_icl_search matrix --execute

icl-eval: ## Build offline Plotly HTML/JSON evaluation from the prepared demo runs
	$(UV_RUN) --extra icl python -m scripts.evaluate_icl

icl-eval-demo: ## Recreate the mock comparison and Plotly dashboard (requires prepared data)
	$(UV_RUN) --extra icl python -m scripts.run_icl_search calibrate --mock
	$(UV_RUN) --extra icl python -m scripts.run_icl_search matrix --mock --matrix configs/icl/experiments/matrix_smoke.json --output analysis/luna_icl/evaluation_demo_runs/easy
	$(UV_RUN) --extra icl python -m scripts.run_icl_search matrix --mock --matrix configs/icl/evaluation/demo_matrix.json --output analysis/luna_icl/evaluation_demo_runs/full
	$(MAKE) icl-eval
