SHELL := cmd.exe
.SHELLFLAGS := /C

PYTHON ?= python
INPUT ?=
RUNTIME_ENV = set SMAP_INTELLIGENCE__EMBEDDINGS__RUNTIME_BACKEND=onnx && set SMAP_INTELLIGENCE__EMBEDDINGS__DEVICE=cpu && set SMAP_INTELLIGENCE__EMBEDDINGS__BATCH_SIZE=24 && set SMAP_INTELLIGENCE__SEMANTIC_PARALLEL_WORKERS=2 && set SMAP_INTELLIGENCE__SEMANTIC_PARALLEL_MIN_MENTIONS=64 && set SMAP_INTELLIGENCE__SEMANTIC_PARALLEL_CHUNK_SIZE=32 &&

.PHONY: install setup db-upgrade doctor run-cosmetics run-beer run-blockchain clean-runtime

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[runtime]"
	if not exist var\data\models mkdir var\data\models

setup: install

db-upgrade:
	$(PYTHON) -m smap.cli.app db-upgrade

doctor:
	$(RUNTIME_ENV) $(PYTHON) -m smap.cli.app doctor

run-cosmetics:
	if "$(INPUT)"=="" (echo INPUT is required. Example: make run-cosmetics INPUT=path\\to\\input.jsonl & exit /b 1)
	$(RUNTIME_ENV) $(PYTHON) -m smap.cli.app run-pipeline "$(INPUT)" --domain-id cosmetics_vn

run-beer:
	if "$(INPUT)"=="" (echo INPUT is required. Example: make run-beer INPUT=path\\to\\input.jsonl & exit /b 1)
	$(RUNTIME_ENV) $(PYTHON) -m smap.cli.app run-pipeline "$(INPUT)" --domain-id beer_vn

run-blockchain:
	if "$(INPUT)"=="" (echo INPUT is required. Example: make run-blockchain INPUT=path\\to\\input.jsonl & exit /b 1)
	$(RUNTIME_ENV) $(PYTHON) -m smap.cli.app run-pipeline "$(INPUT)" --domain-id blockchain_vn

clean-runtime:
	if exist var\app.db del /Q var\app.db
	if exist var\analytics.duckdb del /Q var\analytics.duckdb
	if exist var\data\bronze rmdir /S /Q var\data\bronze
	if exist var\data\silver rmdir /S /Q var\data\silver
	if exist var\data\gold rmdir /S /Q var\data\gold
	if exist var\data\reports rmdir /S /Q var\data\reports
	if exist var\data\insights rmdir /S /Q var\data\insights
	if exist var\data\intelligence\embedding_cache rmdir /S /Q var\data\intelligence\embedding_cache
	if exist var\data\intelligence\vector_index rmdir /S /Q var\data\intelligence\vector_index
	if exist var\data\intelligence\topics rmdir /S /Q var\data\intelligence\topics
	if exist var\data\intelligence\feedback rmdir /S /Q var\data\intelligence\feedback
	if exist var\review rmdir /S /Q var\review
