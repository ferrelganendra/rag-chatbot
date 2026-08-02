"""CLI entry point for ingestion pipeline."""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.pipeline import run_ingestion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    docs, chunks = run_ingestion()
    logging.getLogger(__name__).info(
        f"\u2705 Ingestion complete: {docs} documents \u2192 {chunks} chunks"
    )
