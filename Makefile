BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m
PYTHON_VERSIONS := 3.11.13 3.12.10 3.13.9 3.14.6

.PHONY: format check-formatted check check-matrix tests coverage-branch check-coverage save-coverage docs-llms docs docs-serve examples build wheel-smoke release all prod pre-commit

format:
	@printf "$(BLUE)==>$(RESET) Formatting code with ruff...\n"
	@uv run --extra dev ruff format
	@printf "$(GREEN)✔ Formatting complete.$(RESET)\n"

check-formatted:
	@printf "$(BLUE)==>$(RESET) Checking formatting with ruff format --check...\n"
	@uv run --extra dev ruff format --check
	@printf "$(GREEN)✔ Formatting check complete.$(RESET)\n"

check:
	@printf "$(BLUE)==>$(RESET) Running ruff checks...\n"
	@uv run --extra dev ruff check
	@printf "$(BLUE)==>$(RESET) Type checking with ty...\n"
	@uv run --extra dev ty check
	@printf "$(BLUE)==>$(RESET) Type checking with basedpyright...\n"
	@uv run --extra dev basedpyright
	@printf "$(GREEN)✔ Checking complete.$(RESET)\n"

check-matrix:
	@for version in $(PYTHON_VERSIONS); do \
		short_version=$${version%.*}; \
		printf "$(BLUE)==>$(RESET) Running validation matrix for Python $$version...\n"; \
		uv run --isolated --extra dev --python $$version sh -c \
			"ruff check src/autobench tests && ty check --python-version $$short_version && basedpyright --pythonversion $$short_version src tests && python -m pytest -q" \
			|| exit $$?; \
	done
	@printf "$(GREEN)✔ Matrix checking complete.$(RESET)\n"

tests:
	@printf "$(BLUE)==>$(RESET) Running tests with pytest...\n"
	@uv run --extra dev pytest
	@printf "$(GREEN)✔ Tests complete.$(RESET)\n"

coverage-branch:
	@printf "$(BLUE)==>$(RESET) Running line and branch coverage...\n"
	@uv run --extra dev pytest -p pytest_cov --cov=src/autobench --cov-branch --cov-report=json --cov-fail-under=0 -q
	@printf "$(GREEN)✔ Branch coverage complete. See coverage.json.$(RESET)\n"

check-coverage:
	@printf "$(BLUE)==>$(RESET) Checking 100%% line and branch coverage...\n"
	@set -e; \
		tmp_file=$$(mktemp "$${TMPDIR:-/tmp}/autobench-coverage.XXXXXX"); \
		trap 'rm -f "$$tmp_file"' EXIT; \
		uv run --extra dev pytest -p pytest_cov --cov=src/autobench --cov-branch --cov-report=json:$$tmp_file --cov-fail-under=0 -q; \
		uv run --extra dev python scripts/save_coverage_summary.py --input "$$tmp_file" --check-only
	@printf "$(GREEN)✔ Coverage thresholds satisfied.$(RESET)\n"

save-coverage:
	@printf "$(BLUE)==>$(RESET) Running line and branch coverage...\n"
	@uv run --extra dev pytest -p pytest_cov --cov=src/autobench --cov-branch --cov-report=json --cov-fail-under=0 -q
	@printf "$(BLUE)==>$(RESET) Saving coverage summary to COVERAGE...\n"
	@uv run --extra dev python scripts/save_coverage_summary.py
	@printf "$(GREEN)✔ Coverage summary written to COVERAGE.$(RESET)\n"

docs-llms:
	@printf "$(BLUE)==>$(RESET) Generating LLM documentation bundle...\n"
	@uv run --extra dev python scripts/llms.py --write
	@printf "$(GREEN)✔ LLM documentation bundle generated.$(RESET)\n"

docs:
	@uv run --extra dev python scripts/llms.py --check
	@uv run --extra dev python scripts/llms.py --stage
	@printf "$(BLUE)==>$(RESET) Building Zensical site in strict mode...\n"
	@uv run --extra dev zensical build --clean --strict
	@printf "$(GREEN)✔ Docs build complete.$(RESET)\n"

docs-serve: docs
	@printf "$(BLUE)==>$(RESET) Serving Zensical documentation...\n"
	@uv run --extra dev zensical serve --open

examples:
	@printf "$(BLUE)==>$(RESET) Running offline examples end to end...\n"
	@set -e; root=$$(mktemp -d "$${TMPDIR:-/tmp}/autobench-examples.XXXXXX"); \
		trap 'rm -rf "$$root"' EXIT; \
		for example in minimal basic mid advanced abp_manual abp_concurrent; do \
			uv run autobench validate "examples/$$example/autobench.yaml"; \
			uv run autobench run "examples/$$example/autobench.yaml" --record "$$root/$$example"; \
			uv run autobench replay "$$root/$$example"; \
			uv run autobench report "$$root/$$example"; \
			uv run autobench export "$$root/$$example" --format yaml --path "$$root/$$example-report.yaml"; \
			uv run autobench export "$$root/$$example" --format csv --path "$$root/$$example-runs.csv"; \
		done; \
		uv run autobench dataset generate generator:generate_routing_cases \
			--request examples/generated_dataset/request.yaml \
			--output "$$root/generated-dataset.yaml" \
			--id routing-generated --version v1; \
		uv run python examples/automatic_assets/pydantic_ai_discovery.py --record "$$root/automatic-pydantic-assets"; \
		uv run autobench replay "$$root/automatic-pydantic-assets"; \
		uv run python examples/automatic_assets/custom_sdk_discovery.py --record "$$root/custom-sdk-assets"; \
		uv run autobench replay "$$root/custom-sdk-assets"; \
		uv run python examples/otlp_export/export_record.py "$$root/abp_manual"
	@printf "$(GREEN)✔ Offline examples complete.$(RESET)\n"

build:
	@printf "$(BLUE)==>$(RESET) Building wheel and source distribution...\n"
	@uv build --clear
	@printf "$(GREEN)✔ Build complete.$(RESET)\n"

wheel-smoke: build
	@printf "$(BLUE)==>$(RESET) Smoke testing the built wheel without optional SDKs...\n"
	@set -e; root=$$(mktemp -d "$${TMPDIR:-/tmp}/autobench-wheel.XXXXXX"); \
		trap 'rm -rf "$$root"' EXIT; \
		uv venv "$$root/venv" --python 3.11.13; \
		uv pip install --python "$$root/venv/bin/python" dist/*.whl; \
		"$$root/venv/bin/python" scripts/smoke_wheel.py; \
		"$$root/venv/bin/autobench" instrumentation doctor
	@printf "$(GREEN)✔ Wheel smoke test complete.$(RESET)\n"

all: format check

prod: tests check-coverage check-formatted check docs check-matrix examples

release: prod pre-commit wheel-smoke

pre-commit:
	@printf "$(BLUE)==>$(RESET) Running pre-commit checks...\n"
	@uv run --extra dev pre-commit run --all-files
	@printf "$(GREEN)✔ Pre-commit checks complete.$(RESET)\n"
