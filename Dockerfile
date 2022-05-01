FROM nvcr.io/nvidia/pytorch:21.07-py3
WORKDIR /usr/home
COPY . .
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install -U pip && \
    pip install . && \
    rm -rf ./*