test:
	python -m unittest -v -f test_efficientnet_lite/test*.py

lint:
	pre-commit run --all-files
