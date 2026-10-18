.PHONY: install data augment run evaluate

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

augment:
	python backend/augment.py

run:
	python backend/api.py

evaluate:
	python backend/evaluate.py
