FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .

RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install --no-cache-dir "numpy<2" && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt && \
    python3.11 -m pip uninstall -y numpy && \
    python3.11 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

COPY env_utils.py train.py viewer.py ollama_coach.py ./

EXPOSE 8080 6006

CMD ["python", "train.py"]
