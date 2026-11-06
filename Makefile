.PHONY: install test lint fmt demo clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

demo:
	soar teleop --fake --hz 200 --seconds 1 --record data/demo --task fake_teleop --cameras wrist

clean:
	rm -rf data runs .pytest_cache .ruff_cache .coverage dist build src/*.egg-info
