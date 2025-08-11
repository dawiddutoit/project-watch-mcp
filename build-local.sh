#!/bin/bash

# Build and install project-watch-mcp locally for uvx usage

set -e  # Exit on error

echo "🔨 Building project-watch-mcp..."
uv build

echo "📦 Installing as uv tool..."
uv tool install --force dist/project_watch_mcp-0.1.0-py3-none-any.whl

echo "✅ Testing uvx installation..."
uvx project-watch-mcp --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✨ Successfully installed! You can now use 'uvx project-watch-mcp'"
    echo ""
    echo "📝 Example usage:"
    echo "  uvx project-watch-mcp --repository . --transport stdio"
else
    echo "❌ Installation verification failed"
    exit 1
fi