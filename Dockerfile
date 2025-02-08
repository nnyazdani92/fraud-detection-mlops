FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    curl \
    build-essential \
    python3-pip && \ 
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
        ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

WORKDIR /app

COPY ./src src/
COPY ./MLproject .

RUN mlflow run . --env-manager local

# TODO: Finish Dockerfile
