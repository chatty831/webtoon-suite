#!/bin/bash

# Function to handle cleanup when script is terminated
cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Function to handle timeout after 2 hours
timeout_handler() {
    echo "Timeout reached (2 hours). Shutting down services..."
    cleanup
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
echo "Backend running on PID: $BACKEND_PID (port 9374)"
echo "Frontend running on PID: $FRONTEND_PID (port 8273)"
echo "Services will automatically stop after 2 hours"
echo "Press Ctrl+C to stop both services earlier"

# Start a background timer for 2 hours (7200 seconds)
(sleep 7200 && timeout_handler) &
TIMER_PID=$!

# Wait for either the services to exit or the timer to trigger
wait $BACKEND_PID $FRONTEND_PID $TIMER_PID

# If we reach here, one of the processes has exited
# Clean up any remaining processes
cleanup