include .env

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .
	python -m pip install prodigy -f "https://${PRODIGY_KEY}@download.prodi.gy"

prodigy-basic:
	prodigy duplicate dedup-demo data/recordlinkage_items.jsonl -F recipes/basic.py

prodigy-intermediate:
	prodigy duplicate dedup-demo data/recordlinkage_items.jsonl -F recipes/intermediate.py

prodigy-advanced:
	prodigy duplicate dedup-demo data/nn_trick_items.jsonl -F recipes/intermediate.py

clean:
	rm -rf */__pycache__/*
	rm -rf .ipynb_checkpoints
