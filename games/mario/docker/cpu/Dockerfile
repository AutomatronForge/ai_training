FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        "gym-super-mario-bros==7.4.0" \
        "nes-py==8.2.1" \
        "stable-baselines3[extra]==2.3.2" \
        "opencv-python-headless==4.9.0.80" \
        "shimmy[gym-v21]==1.3.0" \
        "tensorboard" \
        "flask" && \
    pip uninstall -y numpy && \
    pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

COPY env_utils.py env_utils_ram.py train.py train_v0.py train_v3.py train_ram.py viewer.py ollama_coach.py download_pretrained.py entrypoint.py config.py ./

EXPOSE 8080 6006

CMD ["python", "entrypoint.py"]
