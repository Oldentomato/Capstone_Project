FROM python:3.9
COPY /modules /app/modules
COPY /api /app/api
COPY /models /app/models
COPY /utils /app/utils
COPY requirements.txt requirements.txt


RUN apt-get update && \
    python -m pip install --upgrade pip && \
    apt-get -y install libgl1-mesa-glx &&\
    pip install -r requirements.txt

WORKDIR /app

RUN dvc init --no-scm && \
    mkdir model && \
    dvc pull

CMD uvicorn api.api:app
# RUN python ./api/api.py