test:
	@for f in $(shell ls test_efficientnet_lite/test*.py); do \
  		echo $${f};\
		python $${f};\
		done

lint:
	pre-commit run --all-files
