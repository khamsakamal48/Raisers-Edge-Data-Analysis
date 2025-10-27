#!/bin/bash
set -e

echo "Starting Raiser's Edge Data Analysis container..."

# Start cron service
echo "Starting cron service..."
cron

# Display crontab for verification
echo "Installed cron jobs:"
crontab -l

# Keep container running and tail the log file
echo "Container is running. Tailing cron log..."
tail -f /var/log/cron.log
