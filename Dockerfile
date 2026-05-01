FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV VIRTUAL_ENV=/opt/venv

WORKDIR $APP_HOME

# System deps + Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    build-essential \
    libpq-dev \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Optional: make `python` command available
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python deps (cache-friendly)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run app
CMD ["python", "-m", "execution_engine.main"]