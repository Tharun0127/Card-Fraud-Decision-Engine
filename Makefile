# Targets mirror the pipeline stages. Each stage reads the previous stage's artifacts.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: setup data features train decide monitor report test check all smoke clean

setup:
	python3.12 -m venv .venv || py -3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.run_all --stage data

features:
	$(PY) -m src.run_all --stage features

train:
	$(PY) -m src.run_all --stage train

decide:
	$(PY) -m src.run_all --stage decide --stage monitor

report:
	$(PY) -m src.run_all --stage report

test:
	$(PY) -m pytest -q

check:
	$(PY) -m src.reporting.check_numbers

all:
	$(PY) -m src.run_all
	$(PY) -m pytest -q

smoke:
	$(PY) -m src.run_all --synthetic

clean:
	rm -rf data/interim data/processed data/synthetic artifacts/models
