test:
	python -m unittest tests/*.py

lint:
	pre-commit run --all-files
