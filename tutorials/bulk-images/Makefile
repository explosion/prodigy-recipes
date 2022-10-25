install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install prodigy -f "https://${PRODIGY_KEY}@download.prodi.gy"

start: install
	python -m bulk download pets
	python -m bulk download twemoji

