FROM python:3.9-alpine

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir joblib catboost

CMD ["python", "script.py"]
