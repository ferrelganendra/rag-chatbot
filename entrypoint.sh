#!/bin/bash
set -e

echo "Running ingestion..."
python src/ingestion/run.py

echo "Ingestion done. Starting services..."
# Run FastAPI in background, Streamlit in foreground
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
