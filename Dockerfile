FROM nvcr.io/nvidia/pytorch:21.07-py3
WORKDIR /usr/home
COPY . .
RUN apt-get update && \
    apt-get install htop && \
    apt-get -y install git && \
    apt-get -y install ssh && \
    apt-get -y install tmux && \
    pip install -U pip && \
    pip install . && \
    pip install gpustat