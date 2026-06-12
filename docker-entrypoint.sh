#!/bin/bash
set -euo pipefail

echo "=== Meridian Assistant entrypoint ==="
echo "Checking vector store..."

python -c "
from app.data_loader.store import get_document_count
from app.data_loader.pipeline import run_pipeline
from app.logger import setup_logging

setup_logging()
count = get_document_count()
print(f'Vector store chunk count: {count}')
if count == 0:
    print('Running ingestion pipeline...')
    added = run_pipeline()
    print(f'Ingestion complete: {added} chunks added')
else:
    print('Vector store already populated — skipping ingestion')
"

echo "Starting uvicorn..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
