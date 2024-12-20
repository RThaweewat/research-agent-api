# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .env .

RUN mkdir -p src/docs src/data/vectorstore

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["python", "src/main.py"]