# PySpark Coding Challenge - Makefile

VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PYTHON := $(shell command -v python3 2> /dev/null || command -v python 2> /dev/null)

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m

.PHONY: setup test lint lint-fix format format-check type-check coverage \
        run run-custom run-external show-out data-generate data-clean clean clean-all help

all: setup

# Check if virtual environment exists
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(RED)Virtual environment not found!$(NC)"; \
		echo "$(YELLOW)Please run 'make setup' first to create the environment.$(NC)"; \
		exit 1; \
	fi

# Setup project (no editable install; relies on requirements.txt)
setup:
	@echo "$(GREEN)Setting up PySpark Coding Challenge…$(NC)"
	@if [ -z "$(PYTHON)" ]; then \
		echo "$(RED)Error: Python 3 required$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Setup complete$(NC)"
	@echo "$(YELLOW)Run 'make help' for available commands$(NC)"

# -------------------- Dev & QA --------------------
test: check-venv
	@echo "$(GREEN)Running tests…$(NC)"
	@if $(VENV_PYTHON) -m pytest --help | grep -q -- '--cov'; then \
		echo "$(YELLOW)pytest-cov detected — running with coverage$(NC)"; \
		$(VENV_PYTHON) -m pytest tests/ -v \
			--cov=src --cov=scripts \
			--cov-branch \
			--cov-report=term-missing \
			--cov-report=html; \
	else \
		echo "$(YELLOW)pytest-cov not installed — running tests without coverage$(NC)"; \
		$(VENV_PYTHON) -m pytest tests/ -v; \
	fi
	@echo "$(GREEN)✓ Tests completed$(NC)"
	@if [ -d "htmlcov" ]; then \
		echo "$(YELLOW)📊 Coverage report: htmlcov/index.html$(NC)"; \
	fi



lint: check-venv
	@echo "$(GREEN)Ruff lint (check only)…$(NC)"
	$(VENV_PYTHON) -m ruff check src scripts tests

lint-fix: check-venv
	@echo "$(GREEN)Ruff lint (auto-fix, incl. unsafe)…$(NC)"
	$(VENV_PYTHON) -m ruff check --fix --unsafe-fixes src scripts tests

format: check-venv
	@echo "$(GREEN)Black (check only)…$(NC)"
	$(VENV_PYTHON) -m black --check -l 120 src scripts tests

format-check: check-venv
	@echo "$(GREEN)Black (apply formatting)…$(NC)"
	$(VENV_PYTHON) -m black -l 120 src scripts tests

type-check: check-venv
	@echo "$(GREEN)mypy (static type check)…$(NC)"
	MYPYPATH=src $(VENV_PYTHON) -m mypy src

coverage: check-venv
	@if [ ! -f "htmlcov/index.html" ]; then \
		echo "$(RED)No coverage report found. Run 'make test' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Opening coverage report…$(NC)"
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	else \
		echo "$(YELLOW)Please open htmlcov/index.html in your browser$(NC)"; \
	fi

# -------------------- Pipeline runs --------------------
# In-memory demo (no data/ folder needed)
run-demo: check-venv
	@echo "$(GREEN)▶ Running DEMO (in-memory synthetic inputs)…$(NC)"
	OUTDIR="out/demo_training_outputs"; \
	echo "$(YELLOW)Output will be saved under $$OUTDIR$(NC)"; \
	$(VENV_PYTHON) scripts/run.py \
		--demo \
		--out "$$OUTDIR" \
		--max-actions 1000 \
		--lookback-days 365 \
		$(if $(EXPLODE),--explode,) \
		$(if $(SHUF),--shuffle-partitions $(SHUF),)

# Default small mock data (quick smoke test)
run: check-venv
	@echo "$(GREEN)▶ Running with DEFAULT small mock data (30 days, 50 customers, 500 items, batch size 200)…$(NC)"
	@rm -rf data
	OUTDIR="out/default_training_outputs"; \
	$(VENV_PYTHON) scripts/generate_mock_data.py \
		--base data \
		--days 30 \
		--customers 50 \
		--items 500 \
		--batch-size 200; \
	echo "$(YELLOW)Output will be saved under $$OUTDIR$(NC)"; \
	$(VENV_PYTHON) scripts/run.py \
		--out "$$OUTDIR" \
		--max-actions 1000 \
		--lookback-days 365

# Custom run: ask user for mock data params
run-custom: check-venv
	@echo "$(YELLOW)Enter parameters for generating custom mock data.$(NC)"
	@read -p "Number of DAYS of history to generate (e.g. 365): " DAYS; \
	read -p "Number of CUSTOMERS to simulate (e.g. 200): " CUSTOMERS; \
	read -p "Number of ITEMS in catalog (e.g. 2000): " ITEMS; \
	read -p "Batch size for generation (controls JSON file sizes, e.g. 1000): " BATCH; \
	OUTDIR="out/custom_training_outputs"; \
	echo "$(GREEN)▶ Running with CUSTOM mock data (days=$$DAYS, customers=$$CUSTOMERS, items=$$ITEMS, batch=$$BATCH)…$(NC)"; \
	rm -rf data; \
	$(VENV_PYTHON) scripts/generate_mock_data.py \
		--base data \
		--days $$DAYS \
		--customers $$CUSTOMERS \
		--items $$ITEMS \
		--batch-size $$BATCH; \
	echo "$(YELLOW)Output will be saved under $$OUTDIR$(NC)"; \
	$(VENV_PYTHON) scripts/run.py \
		--out "$$OUTDIR" \
		--max-actions 1000 \
		--lookback-days 365

# External run: ask user for dataset paths
run-external: check-venv
	@echo "$(YELLOW)Enter full dataset paths (can be local, HDFS, or S3). Use glob patterns like */*.json if needed.$(NC)"
	@read -p "Impressions path (e.g. /path/to/impr/*.json): " IMPR; \
	read -p "Clicks path (e.g. /path/to/clicks/*.json): " CLICKS; \
	read -p "Add-to-cart path (e.g. /path/to/atc/*.json): " ATC; \
	read -p "Orders path (e.g. /path/to/orders/*.json): " ORDERS; \
	OUTDIR="out/external_training_outputs"; \
	echo "$(GREEN)▶ Running with EXTERNAL datasets…$(NC)"; \
	echo "$(YELLOW)Output will be saved under $$OUTDIR$(NC)"; \
	$(VENV_PYTHON) scripts/run.py \
		--out "$$OUTDIR" \
		--max-actions 1000 \
		--lookback-days 365 \
		--impressions "$$IMPR" \
		--clicks "$$CLICKS" \
		--atc "$$ATC" \
		--orders "$$ORDERS"

# -------------------- Output inspection --------------------
# Show latest parquet output from any run
show-out: check-venv
	@LAST=$$(ls -1dt \
		out/demo_training_outputs/* \
		out/default_training_outputs/* \
		out/custom_training_outputs/* \
		out/external_training_outputs/* 2>/dev/null | head -1); \
	if [ -z "$$LAST" ]; then \
		echo "$(RED)✖ No output runs found in out/demo_training_outputs, out/default_training_outputs, out/custom_training_outputs, or out/external_training_outputs$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)▶ Showing latest Parquet output…$(NC)"; \
	echo "$(YELLOW)Using $$LAST$(NC)"; \
	REL=$${LAST#out/}; \
	$(VENV_PYTHON) scripts/show_output.py --base out --subdir "$$REL" --limit $${ROWS:-10}






# -------------------- Data helpers --------------------
data-generate: check-venv
	@echo "$(GREEN)Generating mock data to ./data (JSON arrays) …$(NC)"
	$(VENV_PYTHON) scripts/generate_mock_data.py

data-clean:
	@echo "$(YELLOW)Removing ./data …$(NC)"
	@rm -rf data

# -------------------- Cleaning --------------------
clean:
	@echo "$(GREEN)Cleaning temporary files…$(NC)"
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage .mypy_cache/ .ruff_cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Removing venv …$(NC)"; \
		rm -rf $(VENV_DIR); \
	fi

# -------------------- Help --------------------
help:
	@echo "$(BLUE)PySpark Coding Challenge$(NC)"
	@echo "$(BLUE)========================$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Development:$(NC)"
	@echo "  $(YELLOW)make setup$(NC)          - Create venv & install requirements"
	@echo "  $(YELLOW)make test$(NC)           - Run tests with coverage (HTML in htmlcov/)"
	@echo "  $(YELLOW)make lint$(NC)           - Ruff lint (check only)"
	@echo "  $(YELLOW)make lint-fix$(NC)       - Ruff lint (auto-fix)"
	@echo "  $(YELLOW)make format$(NC)         - Black (check only)"
	@echo "  $(YELLOW)make format-check$(NC)   - Black (apply formatting)"
	@echo "  $(YELLOW)make type-check$(NC)     - mypy static type checks"
	@echo "  $(YELLOW)make coverage$(NC)       - Open coverage report"
	@echo ""
	@echo "$(GREEN)Pipeline Execution:$(NC)"
	@echo "  $(YELLOW)make run-demo$(NC)       - In-memory demo (no data/ needed)"
	@echo "  $(YELLOW)make run$(NC)            - Run with default small mock dataset"
	@echo "  $(YELLOW)make run-custom$(NC)     - Prompt for mock dataset params at runtime"
	@echo "  $(YELLOW)make run-external$(NC)   - Run on external dataset (paths prompted)"
	@echo "  $(YELLOW)make show-out$(NC)       - Inspect latest Parquet output"
	@echo ""
	@echo "$(GREEN)Data Utilities:$(NC)"
	@echo "  $(YELLOW)make data-generate$(NC)  - Generate mock JSON arrays under ./data/"
	@echo "  $(YELLOW)make data-clean$(NC)     - Remove ./data/"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@echo "  $(YELLOW)make clean$(NC)          - Remove caches/artifacts"
	@echo "  $(YELLOW)make clean-all$(NC)      - Also remove venv"
