#!/bin/bash
echo "=== Starting Super Simple Continuous Training ==="
pkill -f "python3.*train" 2>/dev/null || true
sleep 2
python3 super_simple_continuous.py > super_simple.stdout 2>super_simple.stderr &
PID=$!
echo "Started with PID: $PID"
echo "Checking log file..."
for i in {1..10}; do
    if [ -f "super_simple.log" ] && [ -s "super_simple.log" ]; then
        echo "✓ Log file created and contains content"
        cat super_simple.log
        break
    else
        echo "Waiting for log file... ($i/10)"
        sleep 1
    fi
done
