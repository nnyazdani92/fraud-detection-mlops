FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update -y && \
    apt-get install software-properties-common -y --no-install-recommends && \
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

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY ./src src/
COPY ./MLproject .

EXPOSE 5000

ENTRYPOINT ["mlflow", "run", ".", "--env-manager", "local"]