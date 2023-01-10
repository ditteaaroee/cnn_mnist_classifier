#starting from base image
FROM python:3.10.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

##name training script as the entry point for docker img
ENTRYPOINT [ "python", "-u" , "src/models/predict_model.py"]