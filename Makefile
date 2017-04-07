PYTHON=python
PIP=pip

MODELS=$(wildcard tools/model_*.py)

init:
	$(PIP) install -r requirements.txt -q

test: $(MODELS)
	$(PYTHON) $^

autopep8:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose

flake8:
	flake8 --ignore F401 .

all:
