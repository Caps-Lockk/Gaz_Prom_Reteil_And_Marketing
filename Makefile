.PHONY: install test demo lint help

help:
	@echo 'make install  - установка для разработки'
	@echo 'make test     - тесты'
	@echo 'make demo     - тест на полностью синтетических данных'
	@echo 'make lint     - статическая проверка Python'

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest

demo:
	fuel-elasticity demo --output outputs/demo --clients 1200 --seed 42

lint:
	ruff check src tests
