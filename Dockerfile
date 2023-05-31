FROM python:3.9
COPY /modules /app/modules
COPY /api /app/api
COPY /models /app/models
COPY /utils /app/utils
COPY model.dvc /app/model.dvc
COPY requirements.txt requirements.txt


RUN apt-get update && \
    python -m pip install --upgrade pip && \
    apt-get -y install libgl1-mesa-glx &&\
    pip install -r requirements.txt

WORKDIR /app

RUN mkdir test_image


CMD uvicorn api.api:app --host=0.0.0.0
# RUN python ./api/api.py