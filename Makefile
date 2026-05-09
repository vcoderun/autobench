BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m
PYTHON_VERSIONS := 3.11.13 3.12.10 3.13.9

.PHONY: format check-formatted check check-matrix tests coverage-branch check-coverage save-coverage docs all prod pre-commit

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
		uv run --extra dev --python $$version ruff check src/autobench tests || exit $$?; \
		uv run --extra dev --python $$version ty check --python-version $$short_version || exit $$?; \
		uv run --extra dev --python $$version basedpyright --pythonversion $$short_version src tests || exit $$?; \
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

docs:
	@printf "$(BLUE)==>$(RESET) Building MkDocs site in strict mode...\n"
	@uv run --extra dev mkdocs build --strict
	@printf "$(GREEN)✔ Docs build complete.$(RESET)\n"

all: format check

prod: tests check-coverage format check docs check-matrix

pre-commit:
	@printf "$(BLUE)==>$(RESET) Running pre-commit checks...\n"
	@uv run --extra dev pre-commit run --all-files
	@printf "$(GREEN)✔ Pre-commit checks complete.$(RESET)\n"
