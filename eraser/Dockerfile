# GPU eraser service. Models are NOT baked in: mount them at /models (see README).
#   docker build -t tl-eraser .
#   docker run --gpus all -e TL_TOKEN=$(openssl rand -hex 16) -p 8080:8080 -v /data/models:/models tl-eraser
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir --upgrade pip wheel \
 && python3.10 -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
WORKDIR /app
ENV TL_MODELS=/models TL_PREPARED=/models/prepared
EXPOSE 8080
# Set TL_TOKEN (docker run -e TL_TOKEN=...): the server refuses a public address without it.
CMD ["python3.10", "-m", "eraser.server", "--host", "0.0.0.0", "--port", "8080"]
