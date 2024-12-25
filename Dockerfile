
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app/
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
ENV OPENAI_API_KEY=your-openai-api-key
EXPOSE 5000
CMD ["python", "main.py"]
