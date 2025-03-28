#!/bin/bash

# Activate virtual environment (modify the path if needed)
source venv/Scripts/activate

# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000
