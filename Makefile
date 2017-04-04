PYTHON=python
PIP=pip

MODELS=$(wildcard tools/model_*.py)

init:
	$(PIP) install -r requirements.txt -q

test: $(MODELS)
	$(PYTHON) $<

all:
