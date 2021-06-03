FROM tensorflow/tensorflow:2.4.1
WORKDIR /workspace

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install test requirements
COPY tests/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory
COPY . .
RUN pip3 install --no-cache-dir -e .
