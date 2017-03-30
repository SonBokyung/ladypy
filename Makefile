init:
	pip install -r requirements.txt

simul:
	python simul/parental.py
	python simul/rolemodel.py
	python simul/random.py
