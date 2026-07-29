.PHONY: install install-dev test lint dry-run smoke

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

dry-run:
	pore-pipeline --config configs/final_protocol.json --dry-run

smoke:
	pore-pipeline --config configs/smoke_protocol.json

