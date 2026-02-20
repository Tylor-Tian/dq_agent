PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: bootstrap demo

bootstrap:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .[test]
	mkdir -p artifacts/demo examples
	@test -f examples/orders.csv || echo "missing examples/orders.csv"

demo:
	$(PY) -m dq_agent.cli demo --out artifacts/demo/report.md
