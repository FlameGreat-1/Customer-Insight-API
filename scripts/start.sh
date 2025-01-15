#!/bin/bash

set -e

# Load environment variables
source .env

# Check if PostgreSQL is ready
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_SERVER" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

>&2 echo "PostgreSQL is up - executing command"

# Run database migrations
alembic upgrade head

# Start the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
