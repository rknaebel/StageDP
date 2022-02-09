FROM python:3.7

RUN pip install -U pip setuptools wheel

ADD requirements.txt /app

WORKDIR /app
RUN pip install -r requirements.txt

COPY data data
COPY src src
COPY setup.py README.md ./
RUN pip install -e .

ENTRYPOINT python stagedp/main.py
