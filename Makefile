init:
	pip install -r requirements.txt

simul: dummy
	python simul/parental.py
	python simul/rolemodel.py
	python simul/randomlearn.py

dummy:
