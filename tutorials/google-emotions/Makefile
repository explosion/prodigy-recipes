include .env

download:
	wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
	wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
	wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
	python scripts/prep.py

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install --upgrade prodigy -f "https://${PRODIGY_KEY}@download.prodi.gy"
