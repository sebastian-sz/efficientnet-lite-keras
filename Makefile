test:
	pytest -x test_efficientnet_lite/test_*  # Run all tests except check_output_consistency.py

lint:
	pre-commit run --all-files
