#!/bin/bash

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Export environment variables
export LLAMALITH_CONFIG=config.json

# Run FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
