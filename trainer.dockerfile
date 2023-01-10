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
ENTRYPOINT [ "python", "-u" , "src/models/train_model.py"]


# WORKDIR / app/ -> WORKDIR /app

# COPY requirements.txt requirements.txt

# COPY setup.py setup.py

# COPY src/ src/

# COPY data/ data/

# COPY models/ models/

# RUN  /usr/local/bin/python -m pip install --upgrade pip && \
# pip install -r requirements.txt --no-cache-dir

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]