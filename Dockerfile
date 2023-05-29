FROM python:3.9
COPY /modules /app/modules
COPY /api /app/api
COPY /models /app/models
COPY /utils /app/utils
COPY requirements.txt /app/requirements.txt
COPY /model /app/model

WORKDIR /app/api
RUN apt-get update && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt



CMD uvicorn api:app 
# RUN python ./api/api.py