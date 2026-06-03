FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn mlflow

CMD ["python", "modelling.py"]