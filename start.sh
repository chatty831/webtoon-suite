#!/bin/bash

# Function to handle cleanup when script is terminated
cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set up trap to catch SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend..."
CUDA_VISIBLE_DEVICES=1 uvicorn src.app:app --host 0.0.0.0 --port 9374 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend..."
streamlit run src/frontend/st_app.py --server.port 8273 &
FRONTEND_PID=$!

# Display running services
echo "Backend running on PID: $BACKEND_PID (port 8000)"
echo "Frontend running on PID: $FRONTEND_PID (port 8273)"
echo "Press Ctrl+C to stop both services"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID