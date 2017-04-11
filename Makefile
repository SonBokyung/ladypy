PYTHON=python
PIP=pip

env:
	$(PIP) install -r requirements.txt -q
	flake8 --install-hook git
	git config --bool flake8.strict true
	git config --bool flake8.lazy true

install:
	$(PIP) install .

install-symlink:
	$(PIP) install -e .

remove: uninstall

uninstall:
	$(PIP) uninstall ladypy -y

reinstall: uninstall install

reinstall-symlink: uninstall install-symlink

all: init install
