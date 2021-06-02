test:
	python -m unittest efficientnet_lite/tests/*.py

lint:
	pre-commit run --all-files
