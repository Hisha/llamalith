#!/bin/bash

cd ~/llamalith || exit 1

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Export environment variables
export LLAMALITH_CONFIG=config.json
export PYTHONUNBUFFERED=1

# Run the queue worker
python -u queue_worker.py
