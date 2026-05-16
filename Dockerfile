FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-api.txt && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "nba_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
