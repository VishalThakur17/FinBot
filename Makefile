# FinBot Makefile
# Automates setup, dependency install, and running the app

VENV := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

ifeq ($(OS),Windows_NT)
  PY  := $(VENV)/Scripts/python.exe
  PIP := $(VENV)/Scripts/pip.exe
endif

.PHONY: help venv install run clean

help:
	@echo "Makefile commands:"
	@echo "  make venv     → create virtual environment in $(VENV)"
	@echo "  make install  → install dependencies"
	@echo "  make run      → run backend (server.py)"
	@echo "  make clean    → remove venv and Python cache files"

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip setuptools
	$(PIP) install -r requirements.txt

run:
	$(PY) server.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ *.pyc *.pyo
