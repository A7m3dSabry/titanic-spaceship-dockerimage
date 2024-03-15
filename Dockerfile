FROM python:3.8.10

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . "train.py" /app/
COPY . "test.py" /app/
COPY . "HelloWorld.py" /app/
# COPY . "model.pkl" /app/
COPY . "requirements.txt" /app/



# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
