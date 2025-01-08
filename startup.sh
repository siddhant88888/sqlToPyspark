#!/bin/sh

# Start the FastAPI application
gunicorn backend.backend:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
