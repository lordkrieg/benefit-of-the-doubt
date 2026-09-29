help:
	@echo make install   - install dependencies
	@echo make generate  - build the benchmark
	@echo make evaluate  - evaluate the models, then analyse
	@echo make analyze   - re-run the analysis only
	@echo make status    - progress of generation and evaluation
	@echo make smoke     - evaluate the first 2 passages only
	@echo make mock      - offline dry run with fake models

install:
	uv sync

generate:
	uv run python -m botd.generate

evaluate:
	uv run python -m botd.evaluate

analyze:
	uv run python -m botd.analyze

status:
	uv run python -m botd.generate --status
	uv run python -m botd.evaluate --status

smoke:
	uv run python -m botd.evaluate --limit 2 --no-analyze

mock:
	uv run python -m botd.generate --mock
	uv run python -m botd.evaluate --mock

.PHONY: help install generate evaluate analyze status smoke mock
