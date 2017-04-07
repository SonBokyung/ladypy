PYTHON=python
PIP=pip

MODELS=$(wildcard tools/model_*.py)

init:
	$(PIP) install -r requirements.txt -q

test: $(MODELS)
	$(PYTHON) $^

pep:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose

all:
