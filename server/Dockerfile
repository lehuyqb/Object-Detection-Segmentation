FROM python:3.9-slim

# Build argument for CUDA support
ARG USE_CUDA=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with or without CUDA
RUN if [ "$USE_CUDA" = "true" ] ; then \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Install other requirements
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for weights
RUN mkdir -p weights

EXPOSE 5000

CMD ["python", "app.py"] 