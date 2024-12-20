.PHONY: build run stop clean test

# Docker commands
build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

# Development commands
install:
	pip install -r requirements.txt

dev:
	python src/main.py

test:
	pytest

# Utility commands
logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash 