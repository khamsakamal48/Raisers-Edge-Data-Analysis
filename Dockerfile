# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including cron
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cron \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p Logs "Data Dumps"

# Copy crontab file and set proper permissions
COPY crontab /etc/cron.d/raisers-edge-cron
RUN chmod 0644 /etc/cron.d/raisers-edge-cron && \
    crontab /etc/cron.d/raisers-edge-cron

# Create log file for cron
RUN touch /var/log/cron.log

# Copy and set entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
