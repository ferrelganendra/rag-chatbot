"""CLI entry point for ingestion pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    docs, chunks = run_ingestion()
    print(f"\n✅ Ingestion complete: {docs} documents → {chunks} chunks")
