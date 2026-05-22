FROM python:3.11-slim

# set working directory inside the container
WORKDIR /app

# install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching which makes faster rebuilds)
# --no-cache-dir allows fresh install and is best practice to reduce image size, ensure latest version and cleaner ci/cd pipelines
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of application code
COPY app/ ./app/
COPY data/docs/ ./data/docs/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]