#!/bin/bash

# Test script for Project Watch MCP server
# This script starts the server with default settings for local testing

echo "Starting Project Watch MCP Server..."
echo "Make sure Neo4j is running on localhost:7687"
echo "You can start Neo4j with Docker:"
echo "  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server with the current directory as the repository
uv run project-watch-mcp \
  --repository . \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --neo4j-database neo4j \
  --verbose