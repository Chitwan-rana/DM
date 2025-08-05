#!/bin/sh

echo "Waiting for PostgreSQL..."

while ! nc -z $SQL_HOST $SQL_PORT; do
  sleep 0.1
done

echo "PostgreSQL started"

# Run migrations and collect static files
python manage.py migrate
python manage.py collectstatic --noinput

# Start the Django app with Gunicorn
gunicorn DocumentManager.wsgi:application --bind 0.0.0.0:8000
