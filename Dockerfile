FROM nvcr.io/nvidia/pytorch:21.07-py3
WORKDIR /usr/home
COPY . .
RUN pip install -U pip && \
    pip install .[full] && \
    rm -rf ./*