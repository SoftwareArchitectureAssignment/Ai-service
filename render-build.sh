#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files (nếu cần)
# python manage.py collectstatic --no-input

# Run database migrations (nếu có)
# python manage.py migrate
