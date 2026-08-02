.PHONY: install ingest test eval api ui docker-build clean

PYTHON = python
SRC = src

install:
	pip install -r requirements.txt

ingest:
	$(PYTHON) $(SRC)/ingestion/run.py

test:
	pytest tests/ -v

eval:
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from eval.metrics import evaluate_retrieval; from eval.test_queries import TEST_QUERIES; from retrieval.searcher import Searcher; r = evaluate_retrieval(TEST_QUERIES, Searcher()); print(f'Hit Rate@5: {r[\"hit_rate\"]}, MRR: {r[\"mrr\"]}')"

api:
	uvicorn $(SRC).api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run $(SRC)/ui/app.py

docker-build:
	docker compose build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
