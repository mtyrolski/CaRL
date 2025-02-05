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
