# Image du service local. Le pickle vient de `python train.py` avant le build.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY predict.py ./
COPY models/ridge.pkl ./models/ridge.pkl

RUN pip install --no-cache-dir .

EXPOSE 9696

CMD ["python", "predict.py"]
