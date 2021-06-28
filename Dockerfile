
FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api
COPY exampack /exampack
COPY model.joblib /model.joblib

CMD uvicorn api.app:app --host 0.0.0.0 --port $PORT
