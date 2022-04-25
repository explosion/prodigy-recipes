include .env

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install --upgrade prodigy -f "https://${PRODIGY_KEY}@download.prodi.gy"

download:
	wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/The%20Eiffel%20Tower.npy
	mv "The Eiffel Tower.npy" data

clean:
	rm -rf */__pycache__/*
	rm -rf .ipynb_checkpoints

prodigy-captcha:
	prodigy captcha img-labels-cap -F recipes/captcha.py

prodigy-single:
	prodigy simple-image img-labels-single -F recipes/captcha.py

image-server:
	python -m uvicorn server:app

wipe:
	python -m prodigy drop img-labels-cap