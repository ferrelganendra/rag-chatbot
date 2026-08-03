FROM python:3.12-slim

WORKDIR /app

# Non-root user
RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/documents/ data/documents/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Data dir must be writable by non-root (chroma index)
RUN mkdir -p /app/data/chroma && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
