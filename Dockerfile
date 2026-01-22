FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch first
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the rest (WITHOUT torch/torchvision)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
