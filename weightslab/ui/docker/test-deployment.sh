#!/bin/bash
# Test script to verify Weights Studio deployment in different modes

set -e

DOCKER_COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
CONTAINER_NAME="weights_studio_frontend"

echo "======================================"
echo "Weights Studio Deployment Test"
echo "======================================"

# Function to wait for container
wait_for_container() {
    local container=$1
    local max_attempts=30
    local attempt=0

    echo "Waiting for $container to start..."
    while [ $attempt -lt $max_attempts ]; do
        if docker exec "$container" nginx -t 2>/dev/null; then
            echo "✓ $container is running"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo "✗ $container failed to start"
    return 1
}

# Function to test health endpoint
test_health() {
    local port=$1
    local protocol=${2:-http}

    echo "Testing health endpoint on $protocol://localhost:$port..."

    if [ "$protocol" = "https" ]; then
        if curl -k --silent "https://localhost:$port/health" 2>/dev/null | grep -q "healthy"; then
            echo "✓ HTTPS health check passed"
            return 0
        fi
    else
        if curl --silent "http://localhost:$port/health" 2>/dev/null | grep -q "healthy"; then
            echo "✓ HTTP health check passed"
            return 0
        fi
    fi

    echo "✗ Health check failed"
    return 1
}

# Function to check certificate status
check_certs() {
    if [ -f "../envoy/certs/envoy-server.crt" ] && [ -f "../envoy/certs/envoy-server.key" ]; then
        echo "✓ Certificates found"
        return 0
    else
        echo "✗ Certificates not found (HTTP-only mode)"
        return 1
    fi
}

# Main test flow
echo ""
echo "1. Checking certificate status..."
check_certs
HAVE_CERTS=$?

echo ""
echo "2. Starting Docker stack..."
$DOCKER_COMPOSE_CMD up -d

echo ""
echo "3. Waiting for services to start..."
wait_for_container "$CONTAINER_NAME"

echo ""
echo "4. Checking container configuration..."
docker logs "$CONTAINER_NAME" | grep -E "✓|✗" | head -5

echo ""
echo "5. Testing HTTP(s) endpoint..."
test_health "5173" "http"

echo ""
echo "7. Checking Envoy status..."
if docker exec weights_studio_envoy ps aux | grep -q "envoy"; then
    echo "✓ Envoy proxy is running"
else
    echo "✗ Envoy proxy failed to start"
fi

echo ""
echo "======================================"
echo "Deployment test completed!"
echo "======================================"
echo ""
echo "Access points:"
echo "  HTTP(s): http://localhost:5173"
echo ""
echo "To stop:"
echo "  $DOCKER_COMPOSE_CMD down"
echo ""
echo "To view logs:"
echo "  docker logs weights_studio_frontend"
echo "  docker logs weights_studio_envoy"
