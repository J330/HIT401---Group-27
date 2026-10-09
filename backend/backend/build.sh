#!/usr/bin/env bash
# Runs automatically on every Render deploy.
pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
