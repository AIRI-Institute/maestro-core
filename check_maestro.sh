#!/bin/bash
#
# MAESTRO Core Health Check Script
#

# Verbose mode via environment variable
VERBOSE=${DEBUG:-0}
[ "${D:-0}" -gt 0 ] && VERBOSE=1

# Stop any existing services first
make stop >/dev/null 2>&1 || true
sleep 2

# Ports to check
PORTS=(7732 17231 40631 31611 9681)

# Create test config from llm_config_for_test.json (converted to llm-hub format)
create_test_config() {
    # Read llm_config_for_test.json and extract endpoints
    if [ -f "llm_config_for_test.json" ]; then
        # Use the test config as base, but convert dummy endpoint to use correct descriptor
        cat > llm_config_test.json << 'EOF'
{
  "default_endpoint_key": "dummy-test",
  "warmup": false,
  "endpoints": [
    {
      "key": "dummy-test",
      "descriptor": "mmar_llm.dummy_endpoint.DummyEndpoint",
      "caption": "Dummy Test Endpoint",
      "args": {}
    }
  ]
}
EOF
    else
        log "ERROR: llm_config_for_test.json not found"
        exit 1
    fi
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_cmd() {
    if [ "$VERBOSE" -gt 0 ]; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

# Create converted test config
create_test_config

# ===================================================================
# Prerequisites Check
# ===================================================================
log "=== Prerequisites Check ==="

command -v docker >/dev/null 2>&1 || { log "ERROR: Docker not installed"; exit 1; }
docker info >/dev/null 2>&1 || { log "ERROR: Docker not running"; exit 1; }
log "PASS: Docker is installed and running"

command -v uv >/dev/null 2>&1 || { log "ERROR: uv not installed"; exit 1; }
log "PASS: uv is installed"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -gt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]); then
    log "PASS: Python $PYTHON_VERSION (>= 3.12 required)"
else
    log "ERROR: Python $PYTHON_VERSION is too old (>= 3.12 required)"
    exit 1
fi

# Check ports
for port in "${PORTS[@]}"; do
    if ss -tuln 2>/dev/null | grep -q ":$port "; then
        log "ERROR: Port $port is already in use"
        exit 1
    fi
done
log "PASS: All required ports are available"

# ===================================================================
# Build & Start Services
# ===================================================================
log "=== Build & Start Services ==="

mkdir -p data/maestro
[ -f "data/.env" ] || cp .env.default data/.env
[ -f "data/llm_config.json" ] || cp llm_config_test.json.json data/llm_config.json

run_cmd make build up || { log "ERROR: Failed to build/start services"; exit 1; }
log "PASS: Services built and started"

# Wait for gateway
log "Waiting for services to be ready..."
MAX_WAIT=60
WAIT_TIME=0
READY=false

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s http://localhost:7732/api/health/readiness >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
    echo -n "."
done
echo

if [ "$READY" = true ]; then
    log "PASS: Gateway is ready after ${WAIT_TIME}s"
else
    log "ERROR: Gateway not ready after ${MAX_WAIT}s"
    exit 1
fi

# Give services extra time to initialize
sleep 5

# ===================================================================
# Dummy Track Test
# ===================================================================
log "=== Dummy Track Test ==="

DUMMY_OUTPUT=$(make run-dummy records='dummy=hello dummy=test dummy=exit' 2>&1 || true)

if echo "$DUMMY_OUTPUT" | grep -q "Bot (final)>"; then
    log "PASS: Dummy track test passed"
else
    log "ERROR: Dummy track test failed"
    [ "$VERBOSE" -gt 0 ] && echo "$DUMMY_OUTPUT"
    exit 1
fi

# ===================================================================
# LLM Hub Service Test
# ===================================================================
log "=== LLM Hub Service Test ==="

if curl -s http://localhost:40631/ >/dev/null 2>&1 || nc -z localhost 40631 2>/dev/null; then
    log "PASS: LLM Hub port 40631 is accessible"
else
    log "ERROR: LLM Hub is not accessible on port 40631"
    exit 1
fi

LLM_HUB_LOGS=$(docker compose logs --tail=50 llm-hub 2>/dev/null || echo "")
if echo "$LLM_HUB_LOGS" | grep -q "Server started, listening on 40631"; then
    if echo "$LLM_HUB_LOGS" | grep -q "Endpoints:"; then
        ENDPOINTS_LINE=$(echo "$LLM_HUB_LOGS" | grep "Endpoints:" | tail -1)
        if echo "$ENDPOINTS_LINE" | grep -q "Endpoints: $"; then
            log "WARN: LLM Hub has no endpoints configured"
        else
            log "PASS: LLM Hub loaded configuration"
            [ "$VERBOSE" -gt 0 ] && echo "  $ENDPOINTS_LINE"
        fi
    fi
fi

# ===================================================================
# LLM Config Wizard Test
# ===================================================================
log "=== LLM Config Wizard Test ==="

WIZARD_OUTPUT=$(timeout 30 make run-wizard records='choose_provider=Exit' 2>&1 || true)

if echo "$WIZARD_OUTPUT" | grep -q "Bot (final)>"; then
    FINAL_OUTPUT=$(echo "$WIZARD_OUTPUT" | awk '/^Result:/{flag=1; next} /^Update llm_config.json/{flag=0} flag')
    if echo "$FINAL_OUTPUT" | jq empty >/dev/null 2>&1; then
        log "PASS: Wizard test passed - produced valid JSON"
    else
        log "ERROR: Wizard produced invalid JSON"
        [ "$VERBOSE" -gt 0 ] && echo "$FINAL_OUTPUT"
        exit 1
    fi
else
    log "ERROR: Wizard test did not complete"
    [ "$VERBOSE" -gt 0 ] && echo "$WIZARD_OUTPUT" | tail -20
    exit 1
fi

# ===================================================================
# LLM Chatbot Test
# ===================================================================
log "=== LLM Chatbot Test ==="

CHATBOT_OUTPUT=$(timeout 30 make run-chatbot records='start="Hello test"' 2>&1 || true)

# Check for error conditions
if echo "$CHATBOT_OUTPUT" | grep -q "Bot (state=start)>"; then
    if echo "$CHATBOT_OUTPUT" | grep -q "Not found LLM endpoints"; then
        log "ERROR: LLM endpoints not configured"
        exit 1
    elif echo "$CHATBOT_OUTPUT" | grep -qiE "(authentication error|invalid api key|unauthorized|401|403)"; then
        log "ERROR: LLM API key is invalid or authentication failed"
        exit 1
    elif echo "$CHATBOT_OUTPUT" | grep -q "Request:"; then
        log "PASS: LLM chatbot test passed - got response from dummy endpoint"
    else
        RESPONSE_TEXT=$(echo "$CHATBOT_OUTPUT" | grep "Bot (state=start)>" | tail -1 | sed 's/.*Bot (state=start)> //' | head -c 100)
        if [ -n "$RESPONSE_TEXT" ] && [ "$RESPONSE_TEXT" != "" ]; then
            log "PASS: LLM chatbot test passed - got response from LLM"
        else
            log "ERROR: LLM returned empty or invalid response"
            [ "$VERBOSE" -gt 0 ] && echo "$CHATBOT_OUTPUT"
            exit 1
        fi
    fi
elif echo "$CHATBOT_OUTPUT" | grep -q "Bot (final)>"; then
    FINAL_MSG=$(echo "$CHATBOT_OUTPUT" | grep "Bot (final)>" | head -1)
    log "ERROR: LLM test failed - bot in error state: $FINAL_MSG"
    exit 1
else
    log "ERROR: LLM chatbot test failed - no valid response"
    [ "$VERBOSE" -gt 0 ] && echo "$CHATBOT_OUTPUT"
    exit 1
fi

# ===================================================================
# Summary
# ===================================================================
echo ""
log "=== Health Check Summary ==="
log "All tests passed!"
log "MAESTRO Core is working correctly."

# Cleanup - stop services
log "Stopping services..."
make stop >/dev/null 2>&1 || true
log "Done."
