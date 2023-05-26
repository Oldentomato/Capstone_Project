FROM python:3.9
COPY /modules /app/modules
COPY /api /app/api
COPY /model /app/model
COPY /models /app/models
COPY /utils /app/utils
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD tail -f /dev/null
# RUN python ./api/api.py