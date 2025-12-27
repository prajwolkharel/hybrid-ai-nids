# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nfstream with system deps for packet capture
RUN apt-get update && apt-get install -y libpcap-dev && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command: run dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
