# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 appuser
# Fix permissions
RUN chown -R appuser:appuser /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Note: Ensure .dockerignore blocks .venv and other unnecessary large files
COPY . .

# Fix permissions again after copying
RUN chown -R appuser:appuser /app
# Switch to the non-root user
USER appuser

# Expose the designated port for Hugging Face Spaces
EXPOSE 7860

# Extract the vector DB on startup and run the FastAPI app via Uvicorn on port 7860
CMD unzip -o chroma_db.zip -d rag/ && uvicorn api.main:app --host 0.0.0.0 --port 7860
