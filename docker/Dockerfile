FROM tensorflow/tensorflow:2.4.1
WORKDIR /workspace

# Upgrade pip
RUN /usr/bin/python3 -m pip install --upgrade pip

# Install requirements:
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install test requirements
COPY efficientnet_lite/tests/requirements.txt .
RUN pip3 install -r requirements.txt

# Install repo for imports to work properly
COPY setup.py .
RUN pip3 install -e .
