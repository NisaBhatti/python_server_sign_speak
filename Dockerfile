FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8080
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --preload --max-requests 30 --max-requests-jitter 5