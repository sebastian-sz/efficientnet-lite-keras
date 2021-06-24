ARG IMAGE_TAG=2.5.0

FROM tensorflow/tensorflow:${IMAGE_TAG}

# Install package from repository
WORKDIR /tmp/
COPY . .
RUN pip3 install --no-cache-dir . && rm -r /tmp/*

WORKDIR /workspace/
