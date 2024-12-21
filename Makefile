install:
	pip install -r requirements.txt
lint:
	cat docker/backend.Dockerfile | docker run --rm -i hadolint/hadolint
	cat docker/frontend.Dockerfile | docker run --rm -i hadolint/hadolint
	flake8 . --max-line-length=120
	pylint . --max-line-length=120 --disable="C0103,C0114,C0115,R0801,R0913,R0914,R0917,E1136,E0213,W0707"


build:
	docker-compose build

run:
	docker-compose up