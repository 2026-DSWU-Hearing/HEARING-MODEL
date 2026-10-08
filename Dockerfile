FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    TFHUB_CACHE_DIR=/app/.cache/tfhub

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/.cache/tfhub

EXPOSE 8765
CMD ["python", "run.py"]
