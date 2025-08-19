#!/bin/bash

# Script to run all incremental indexing tests with proper categorization
# Usage: ./run_incremental_tests.sh [unit|integration|performance|edge|all]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test category to run (default: all)
CATEGORY=${1:-all}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Incremental Indexing Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run tests with coverage
run_tests() {
    local test_files="$1"
    local description="$2"
    
    echo -e "${YELLOW}Running $description...${NC}"
    
    if python -m pytest $test_files -v --tb=short --cov=src/project_watch_mcp --cov-report=term-missing --cov-report=html --cov-report=json; then
        echo -e "${GREEN}✅ $description PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $description FAILED${NC}"
        return 1
    fi
}

# Check if Neo4j is available for integration tests
check_neo4j() {
    if [ -z "$NEO4J_URI" ]; then
        echo -e "${YELLOW}⚠️  Neo4j not configured. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to run integration tests${NC}"
        echo -e "${YELLOW}   Example: export NEO4J_URI=bolt://localhost:7687${NC}"
        return 1
    fi
    return 0
}

# Track overall success
overall_success=true

# Run tests based on category
case $CATEGORY in
    unit)
        echo -e "${BLUE}Running Unit Tests Only${NC}"
        echo ""
        run_tests "tests/unit/test_incremental_indexing.py tests/unit/test_cli_incremental.py" "Unit Tests" || overall_success=false
        ;;
    
    integration)
        echo -e "${BLUE}Running Integration Tests Only${NC}"
        echo ""
        if check_neo4j; then
            run_tests "tests/integration/server/test_incremental_indexing.py" "Integration Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping integration tests (Neo4j not configured)${NC}"
        fi
        ;;
    
    performance)
        echo -e "${BLUE}Running Performance Tests Only${NC}"
        echo ""
        if check_neo4j; then
            run_tests "tests/integration/performance/test_incremental_performance.py -m performance" "Performance Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping performance tests (Neo4j not configured)${NC}"
        fi
        ;;
    
    edge)
        echo -e "${BLUE}Running Edge Case Tests Only${NC}"
        echo ""
        if check_neo4j; then
            run_tests "tests/integration/test_incremental_edge_cases.py" "Edge Case Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping edge case tests (Neo4j not configured)${NC}"
        fi
        ;;
    
    all)
        echo -e "${BLUE}Running All Tests${NC}"
        echo ""
        
        # Unit tests (always run)
        echo -e "${BLUE}Step 1/4: Unit Tests${NC}"
        run_tests "tests/unit/test_incremental_indexing.py tests/unit/test_cli_incremental.py" "Unit Tests" || overall_success=false
        echo ""
        
        # Integration tests (if Neo4j available)
        echo -e "${BLUE}Step 2/4: Integration Tests${NC}"
        if check_neo4j; then
            run_tests "tests/integration/server/test_incremental_indexing.py" "Integration Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping (Neo4j not configured)${NC}"
        fi
        echo ""
        
        # Performance tests (if Neo4j available)
        echo -e "${BLUE}Step 3/4: Performance Tests${NC}"
        if check_neo4j; then
            run_tests "tests/integration/performance/test_incremental_performance.py" "Performance Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping (Neo4j not configured)${NC}"
        fi
        echo ""
        
        # Edge case tests (if Neo4j available)
        echo -e "${BLUE}Step 4/4: Edge Case Tests${NC}"
        if check_neo4j; then
            run_tests "tests/integration/test_incremental_edge_cases.py" "Edge Case Tests" || overall_success=false
        else
            echo -e "${YELLOW}Skipping (Neo4j not configured)${NC}"
        fi
        ;;
    
    *)
        echo -e "${RED}Invalid category: $CATEGORY${NC}"
        echo "Usage: $0 [unit|integration|performance|edge|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"

# Generate coverage report summary
if [ -f "coverage.json" ]; then
    echo -e "${BLUE}Coverage Summary:${NC}"
    python -c "
import json
with open('coverage.json') as f:
    data = json.load(f)
    total = data['totals']
    print(f'  Overall Coverage: {total[\"percent_covered\"]:.1f}%')
    print(f'  Lines Covered: {total[\"covered_lines\"]}/{total[\"num_statements\"]}')
    
    # Show coverage for key files
    files = data['files']
    key_files = [
        'src/project_watch_mcp/neo4j_rag.py',
        'src/project_watch_mcp/core/initializer.py',
        'src/project_watch_mcp/cli.py'
    ]
    print(f'\\n  Key Files:')
    for f in key_files:
        if f in files:
            cov = files[f]['summary']['percent_covered']
            print(f'    {f.split(\"/\")[-1]}: {cov:.1f}%')
" 2>/dev/null || echo "  (Coverage data not available)"
fi

echo -e "${BLUE}========================================${NC}"

# Final status
if [ "$overall_success" = true ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    echo -e "${GREEN}   Coverage report: htmlcov/index.html${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check output above.${NC}"
    exit 1
fi