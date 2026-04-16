# Dockerfile
# Build: docker build -t sotay-rag .
# Run:   docker run -p 8000:8000 --env-file .env sotay-rag

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by unstructured for PDF parsing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# NOTE: The vectorstore must be pre-built and committed to the image.
# Run `python ingest.py` locally first, then build the Docker image.
# The ./vectorstore/ directory is included in the build context.

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
