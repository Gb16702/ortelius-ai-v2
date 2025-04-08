.PHONY: dev install

install:
	uv pip install -e .

dev:
	uvicorn app.main:app --reload
