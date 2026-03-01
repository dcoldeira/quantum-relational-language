FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the platform needs at runtime
COPY src/ ./src/
COPY qai/ ./qai/
COPY docs/api-reference.md ./docs/api-reference.md

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "qai.api:app", "--host", "0.0.0.0", "--port", "8000"]
