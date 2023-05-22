FROM python:3.9
COPY /yolomodules /app/yolomodules
COPY /api /app/api
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD tail -f /dev/null
# RUN python ./api/api.py