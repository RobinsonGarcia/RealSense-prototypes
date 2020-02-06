FROM registry-proxy.petrobras.com.br/ubuntu:16.04

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk 

RUN pip3 install --upgrade pip

COPY ./requirements.txt /workspace/requirements.txt

RUN pip3 install -r requirements.txt

CMD ["python"]