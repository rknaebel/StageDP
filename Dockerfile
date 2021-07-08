FROM python:3.7

RUN pip install -U pip setuptools wheel

ADD requirements.txt /opt/stage-dp/

WORKDIR /opt/stage-dp
RUN pip install -r requirements.txt

COPY data /opt/stage-dp/data
COPY src /opt/stage-dp/src
COPY tests /opt/stage-dp/tests
COPY setup.py README.md /opt/stage-dp/
RUN pip install -e .

ENTRYPOINT ["python", "src/stagedp/parser_wrapper.py"]

