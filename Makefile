PYTHON=python
PIP=pip

MODELS=$(wildcard tools/model_*.py)

init:
	$(PIP) install -r requirements.txt -q
	flake8 --install-hook git
	git config --bool flake8.strict true
	git config --bool flake8.lazy true

test: $(MODELS)
	$(PYTHON) $^

install:
	$(PIP) install .

install-symlink:
	$(PIP) install -e .

remove: uninstall

uninstall:
	$(PIP) uninstall elg -y

reinstall: uninstall install

reinstall-symlink: uninstall install-symlink

all: init install
