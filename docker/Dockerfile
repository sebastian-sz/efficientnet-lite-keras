FROM tensorflow/tensorflow:2.4.1
WORKDIR /workspace

# Copy files to install package
COPY README.md .
COPY requirements.txt .
COPY efficientnet_lite efficientnet_lite
COPY setup.py .

# Install:
RUN /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r efficientnet_lite/tests/requirements.txt && \
    pip3 install --no-cache-dir -e .
