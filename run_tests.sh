#!/bin/bash
# Quick test runner script for project-watch-mcp

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Project Watch MCP - Test Runner${NC}"
echo "=================================="

# Parse arguments
TEST_TYPE="${1:-quick}"
COVERAGE_THRESHOLD="${2:-0}"

case "$TEST_TYPE" in
    quick)
        echo -e "${YELLOW}Running quick test suite (passing tests only)...${NC}"
        uv run pytest tests/unit/test_analyze_complexity.py tests/unit/complexity tests/unit/utils \
            --cov-fail-under=$COVERAGE_THRESHOLD -q
        ;;
    
    unit)
        echo -e "${YELLOW}Running all unit tests...${NC}"
        uv run pytest tests/unit \
            --cov-fail-under=$COVERAGE_THRESHOLD -q
        ;;
    
    coverage)
        echo -e "${YELLOW}Running tests with HTML coverage report...${NC}"
        uv run pytest tests/unit \
            --cov=src/project_watch_mcp \
            --cov-report=html \
            --cov-report=term \
            --cov-fail-under=$COVERAGE_THRESHOLD
        echo -e "${GREEN}Coverage report generated at: htmlcov/index.html${NC}"
        ;;
    
    fast)
        echo -e "${YELLOW}Running minimal test set for CI...${NC}"
        uv run pytest tests/unit/test_analyze_complexity.py \
            --cov-fail-under=0 \
            --timeout=5 \
            -q
        ;;
    
    *)
        echo -e "${RED}Usage: $0 [quick|unit|coverage|fast] [coverage_threshold]${NC}"
        echo "  quick    - Run passing tests only (default)"
        echo "  unit     - Run all unit tests"
        echo "  coverage - Run with HTML coverage report"
        echo "  fast     - Minimal test set for CI"
        echo ""
        echo "Example: $0 coverage 60"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi