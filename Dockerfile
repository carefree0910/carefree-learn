FROM nvcr.io/nvidia/pytorch:22.09-py3
WORKDIR /usr/home
COPY . .
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2 && \
    pip install -U pip && \
    pip install .[full] && \
    rm -rf ./*