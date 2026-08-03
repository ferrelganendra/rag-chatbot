#!/bin/bash
set -e

# If a command was passed (docker-compose per-service `command:`), run only that.
if [ $# -gt 0 ]; then
  exec "$@"
fi

echo ">>> DocQ — Ingestion Start"
python src/ingestion/run.py
echo ">>> Ingestion Complete"

echo ">>> Starting FastAPI on :8000"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

echo ">>> Starting Streamlit on :8501"
exec streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
