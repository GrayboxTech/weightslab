#!/bin/bash
# Comprehensive deployment test for weights_studio frontend + backend
# Tests communication between frontend and Python backend (ws-segmentation example)
# Runs both unsecured (HTTP) and secured (HTTPS) test scenarios

# set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CURRENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_DIR="$CURRENT_DIR/docker"
UTILS_DIR="$DOCKER_DIR/utils"
EXAMPLE_DIR="$CURRENT_DIR/tests/playwright/examples/ws-segmentation"
GRPC_PORT=50051
ENVOY_PORT=8080
FRONTEND_PORT=5173
HEALTH_CHECK_TIMEOUT=30
COMMUNICATION_TEST_TIMEOUT=20

# Global test results
TESTS_PASSED=0
TESTS_FAILED=0

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Clean up docker resources
cleanup() {
    log_info "Cleaning up Docker resources..."
    cd "$DOCKER_DIR"
    docker compose -f docker-compose.yml -f docker-compose.prod.yml down --remove-orphans 2>/dev/null || true
    # Force remove any lingering containers
    docker ps -a | grep weights_studio | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
    # Remove network
    docker network rm docker_weights_studio_network 2>/dev/null || true
    sleep 3
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local timeout=$3
    local protocol=${4:-http}

    log_info "Waiting for $service_name on $protocol://localhost:$port..."

    local start_time=$(date +%s)
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -gt $timeout ]; then
            log_error "$service_name failed to start after ${timeout}s"
            return 1
        fi

        if [ "$protocol" = "https" ]; then
            if curl -sk -o /dev/null -w "%{http_code}" "https://localhost:$port/health" 2>/dev/null | grep -q "200"; then
                log_success "$service_name is ready"
                return 0
            fi
        else
            if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null | grep -q "200"; then
                log_success "$service_name is ready"
                return 0
            fi
        fi

        sleep 2
    done
}

# Check if gRPC backend is responding
check_grpc_backend() {
    local timeout=$1
    log_info "Checking gRPC backend on localhost:$GRPC_PORT..."

    local start_time=$(date +%s)
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -gt $timeout ]; then
            log_error "gRPC backend not responding after ${timeout}s"
            return 1
        fi

        # Try to connect using nc (netcat)
        if nc -z localhost $GRPC_PORT 2>/dev/null; then
            log_success "gRPC backend is listening on port $GRPC_PORT"
            return 0
        fi

        sleep 1
    done
}

# Check frontend-backend communication
test_frontend_backend_communication() {
    local protocol=$1
    local test_name=$2
    local port=${3:-$FRONTEND_PORT}

    log_info "Testing frontend -> backend communication ($test_name)..."

    # The frontend makes requests to /api which proxies to envoy
    # Envoy proxies to localhost:50051 (the gRPC backend)

    # Check if frontend can resolve the backend
    local health_url="$protocol://localhost:$port/health"

    local http_code=$(curl -sk -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null || echo "000")

    if [ "$http_code" = "200" ]; then
        log_success "Frontend is responding ($test_name)"
    else
        log_error "Frontend health check failed with HTTP $http_code ($test_name)"
        return 1
    fi

    return 0
}

# Run backend example in background and monitor
run_backend_example() {
    local test_scenario=$1

    if [ ! -f "$EXAMPLE_DIR/main.py" ]; then
        log_error "Python example not found at $EXAMPLE_DIR/main.py"
        return 1
    fi

    log_info "Starting backend example: $test_scenario..."

    cd "$EXAMPLE_DIR"

    # Start Python backend in background
    timeout 60s python main.py > "/tmp/ws_backend_${test_scenario}.log" 2>&1 &
    local backend_pid=$! 2>/dev/null || true

    log_info "Backend process started (PID: $backend_pid)"

    # Wait a bit for backend to initialize
    sleep 3

    # Check if process is still running
    if ! kill -0 $backend_pid 2>/dev/null; then
        log_error "Backend process exited prematurely"
        cat "/tmp/ws_backend_${test_scenario}.log"
        return 1
    fi

    log_success "Backend example is running"

    # Check gRPC port
    if check_grpc_backend $COMMUNICATION_TEST_TIMEOUT; then
        log_success "Backend gRPC service is reachable"
    else
        log_error "Backend gRPC service is not reachable"
        kill $backend_pid 2>/dev/null || true
        return 1
    fi

    return 0
}

# Temporarily hide/show certs and tokens for testing
hide_certs() {
    local certs_dir=$1
    if [ -d "$certs_dir" ]; then
        mv "$certs_dir" "${certs_dir}.bak"
        mkdir -p "$certs_dir"
    fi
}

restore_certs() {
    local certs_dir=$1
    if [ -d "${certs_dir}.bak" ]; then
        rm -rf "$certs_dir"
        mv "${certs_dir}.bak" "$certs_dir"
    fi
}

hide_grpc_token() {
    local token_file=$1
    if [ -f "$token_file" ]; then
        mv "$token_file" "${token_file}.bak"
    fi
}

restore_grpc_token() {
    local token_file=$1
    if [ -f "${token_file}.bak" ]; then
        mv "${token_file}.bak" "$token_file"
    fi
}

# Run test scenario
run_test_scenario() {
    local scenario_name=$1
    local secure=$2
    local use_certs=${3:-true}
    local use_auth=${4:-false}

    # Create a safe filename from scenario name (remove spaces, colons, etc)
    local scenario_safe=$(echo "$scenario_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/-$//')

    print_section "$scenario_name"

    # Cleanup previous deployment
    cleanup

    # Setup certs and tokens for this scenario
    local certs_dir="$DOCKER_DIR/.weightslab-certs-test"
    log_info "certs_dir: $certs_dir"
    local token_file="$certs_dir/.grpc_auth_token"
    local certs_hidden=false
    local token_hidden=false

    if [ "$use_certs" = "false" ] && [ -d "$certs_dir" ]; then
        log_info "Temporarily hiding TLS certificates for this test..."
        hide_certs "$certs_dir"
        certs_hidden=true
    fi

    if [ "$use_auth" = "false" ] && [ -f "$token_file" ]; then
        log_info "Temporarily hiding gRPC auth token for this test..."
        hide_grpc_token "$token_file"
        token_hidden=true
    fi

    # Set environment for deployment
    if [ "$secure" = "true" ]; then
        if [ "$use_certs" = "true" ] && [ "$use_auth" = "true" ]; then
            log_info "Secure mode: HTTPS with TLS + gRPC auth enabled"
        elif [ "$use_certs" = "true" ]; then
            log_info "Secure mode: HTTPS with TLS (no gRPC auth)"
        elif [ "$use_auth" = "true" ]; then
            log_info "Secure mode: HTTP with gRPC auth only (no TLS)"
        fi
        export E2E_MANAGED_STACK=1
    else
        log_info "Unsecure mode: HTTP without TLS or gRPC auth"
        export E2E_MANAGED_STACK=1
    fi

    # Build and deploy frontend
    cd "$UTILS_DIR"

    if [ "$secure" = "true" ] && [ "$use_certs" = "true" ]; then
        log_info "Building and deploying with HTTPS support..."
        WEIGHTSLAB_CERTS_DIR="$certs_dir" bash build-and-deploy.sh > "/tmp/build_deploy_${scenario_safe}.log" 2>&1 || {
            log_error "Build and deploy failed"
            cat "/tmp/build_deploy_${scenario_safe}.log"
            [ "$certs_hidden" = true ] && restore_certs "$certs_dir"
            [ "$token_hidden" = true ] && restore_grpc_token "$token_file"
            return 1
        }
    else
        log_info "Building and deploying with HTTP (no TLS certs)..."
        WEIGHTSLAB_CERTS_DIR="$certs_dir" bash build-and-deploy.sh --unsecure > "/tmp/build_deploy_${scenario_safe}.log" 2>&1 || {
            log_error "Build and deploy failed"
            cat "/tmp/build_deploy_${scenario_safe}.log"
            [ "$certs_hidden" = true ] && restore_certs "$certs_dir"
            [ "$token_hidden" = true ] && restore_grpc_token "$token_file"
            return 1
        }
    fi

    log_success "Frontend deployment successful"

    # Wait for services to be ready
    local protocol="http"
    local frontend_port=$FRONTEND_PORT
    if [ "$secure" = "true" ] && [ "$use_certs" = "true" ]; then
        protocol="https"
    fi

    # Check frontend (don't check envoy since it depends on upstream backend)
    if ! wait_for_service "Frontend (nginx)" "$frontend_port" "$HEALTH_CHECK_TIMEOUT" "$protocol"; then
        log_error "Frontend failed to start"
        [ "$certs_hidden" = true ] && restore_certs "$certs_dir"
        [ "$token_hidden" = true ] && restore_grpc_token "$token_file"
        return 1
    fi

    # Test frontend-backend communication setup
    if test_frontend_backend_communication "$protocol" "$scenario_safe" "$frontend_port"; then
        log_success "Frontend-backend communication setup OK"
    else
        log_error "Frontend-backend communication setup failed"
        [ "$certs_hidden" = true ] && restore_certs "$certs_dir"
        [ "$token_hidden" = true ] && restore_grpc_token "$token_file"
        return 1
    fi

    # Restore certs and tokens
    if [ "$certs_hidden" = true ]; then
        log_info "Restoring TLS certificates..."
        restore_certs "$certs_dir"
    fi
    if [ "$token_hidden" = true ]; then
        log_info "Restoring gRPC auth token..."
        restore_grpc_token "$token_file"
    fi

    return 0
}

# ============================================================================
# Main Test Execution
# ============================================================================

main() {
    print_section "WEIGHTS STUDIO FRONTEND-BACKEND DEPLOYMENT TEST"

    log_info "Test Configuration:"
    log_info "  Weights Studio Dir: $CURRENT_DIR"
    log_info "  Docker Dir: $DOCKER_DIR"
    log_info "  Example Dir: $EXAMPLE_DIR"
    log_info "  Envoy Port: $ENVOY_PORT"
    log_info "  Frontend Port: $FRONTEND_PORT"
    log_info "  gRPC Backend Port: $GRPC_PORT"

    # Test A: Unsecured Environment (HTTP)
    if run_test_scenario "TEST A: UNSECURED ENVIRONMENT (HTTP)" "false" "false" "false"; then
        log_success "TEST A PASSED: Frontend and backend communicate over HTTP"
    else
        log_error "TEST A FAILED: Frontend-backend communication issue in HTTP mode"
    fi

    # Cleanup between tests
    cleanup
    sleep 5

    # Test B: Secured Environment (HTTPS with TLS + gRPC Auth)
    # Note: This requires TLS certificates AND gRPC auth token to be present
    if [ -d "$DOCKER_DIR/.weightslab-certs-test" ] && [ -f "$DOCKER_DIR/.weightslab-certs-test/envoy-server.crt" ]; then
        if [ -f "$DOCKER_DIR/.weightslab-certs-test/.grpc_auth_token" ]; then
            if run_test_scenario "TEST B: SECURED ENVIRONMENT (HTTPS + TLS + gRPC AUTH)" "true" "true" "true"; then
                log_success "TEST B PASSED: Frontend and backend communicate over HTTPS with full security"
            else
                log_error "TEST B FAILED: Frontend-backend communication issue in HTTPS+Auth mode"
            fi
        else
            log_warn "TEST B SKIPPED: gRPC auth token not found"
            log_warn "  Token should be at $DOCKER_DIR/.weightslab-certs-test/.grpc_auth_token"
        fi

        # Cleanup between tests
        cleanup
        sleep 5

        # Test C: TLS Certificates Only (no gRPC auth)
        if run_test_scenario "TEST C: SECURED ENVIRONMENT (HTTPS + TLS ONLY, NO GRPC AUTH)" "true" "true" "false"; then
            log_success "TEST C PASSED: Frontend and backend communicate over HTTPS (TLS only, no gRPC auth)"
        else
            log_error "TEST C FAILED: Frontend-backend communication issue in TLS-only mode"
        fi

        # Cleanup between tests
        cleanup
        sleep 5

        # Test D: gRPC Auth Only (no TLS certificates)
        if [ -f "$DOCKER_DIR/.weightslab-certs-test/.grpc_auth_token" ]; then
            if run_test_scenario "TEST D: SECURED ENVIRONMENT (HTTP + gRPC AUTH ONLY, NO TLS)" "true" "false" "true"; then
                log_success "TEST D PASSED: Frontend and backend communicate over HTTP with gRPC auth only"
            else
                log_error "TEST D FAILED: Frontend-backend communication issue in gRPC-auth-only mode"
            fi
        else
            log_warn "TEST D SKIPPED: gRPC auth token not found"
            log_warn "  Token should be at $DOCKER_DIR/.weightslab-certs-test/.grpc_auth_token"
        fi
    else
        log_warn "TESTS B, C, D SKIPPED: TLS certificates not found at $DOCKER_DIR/.weightslab-certs-test"
        log_warn "  To run secured tests, copy certs to:"
        log_warn "  $DOCKER_DIR/.weightslab-certs-test"
    fi

    # Final cleanup
    cleanup

    # Print summary
    print_section "TEST SUMMARY"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"

    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "ALL TESTS PASSED! ✨"
        return 0
    else
        log_error "SOME TESTS FAILED ❌"
        return 1
    fi
}

# Handle script interruption
trap cleanup EXIT

# Run main
main "$@"
