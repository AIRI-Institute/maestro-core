#!/bin/bash
# Demonstration of API mismatch error using Docker
# This script:
# 1. Builds a Docker image with a server that expects config: str
# 2. Starts the server container
# 3. Runs a client with incompatible API (config: dict)
# 4. Shows the error that occurs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "API Mismatch Error Demonstration"
echo "=========================================="
echo ""

# Build Docker image for server
echo "1. Building Docker image with server..."
echo "   Server API: interpret(*, config: str, ...)"
echo ""

docker build -f Dockerfile.server -t mmar-ptag-server:demo -q . 2>/dev/null || \
    docker build -f Dockerfile.server -t mmar-ptag-server:demo .

echo ""
echo "2. Starting server container..."
echo "   Port: 50051"
echo ""

# Remove any existing container
docker rm -f mmar-ptag-server 2>/dev/null || true

# Start server container
docker run -d --name mmar-ptag-server \
    --network host \
    mmar-ptag-server:demo

# Wait for server to start
echo "   Waiting for server to start..."
sleep 3

# Check if server is running
if ! docker ps | grep -q mmar-ptag-server; then
    echo "   ERROR: Server failed to start!"
    docker logs mmar-ptag-server
    exit 1
fi

echo "   Server started."
echo ""

# Run INCOMPATIBLE client (sends dict, server expects str)
echo "3. Running INCOMPATIBLE client..."
echo "   Client API: interpret(*, config: dict, ...)"
echo "   This WILL FAIL with validation error!"
echo ""
echo "=========================================="
echo ""

# Install dependencies if needed and run incompatible client
uv run --with grpcio python examples/client_incompatible.py || \
    python -c "import sys; sys.path.insert(0, '.'); exec(open('examples/client_incompatible.py').read())"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo ""

# Show server logs
echo "Server logs showing the error:"
echo "-----------------------------------"
docker logs mmar-ptag-server 2>&1 | tail -20

echo ""
echo "4. Cleaning up..."
docker rm -f mmar-ptag-server >/dev/null 2>&1

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Demo completed - API mismatch error shown above"
else
    echo "✗ Demo failed"
fi

exit $EXIT_CODE
