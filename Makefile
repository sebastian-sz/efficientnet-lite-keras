test:
	pytest -x test_efficientnet_lite/test*.py

lint:
	pre-commit run --all-files
