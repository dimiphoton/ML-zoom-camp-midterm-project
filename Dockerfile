# Image du service. Le pickle vient de `python train.py` avant le build.
# En local le port publié est 9696. Sur Render, predict.py lit PORT.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY predict.py ./
COPY models/ridge.pkl ./models/ridge.pkl

RUN pip install --no-cache-dir .

EXPOSE 9696

CMD ["python", "predict.py"]
