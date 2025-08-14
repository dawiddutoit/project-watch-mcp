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
    echo "🔄 Checking for running MCP server processes..."
    
    # Find and kill any existing project-watch-mcp processes
    PIDS=$(pgrep -f "project-watch-mcp" 2>/dev/null || true)
    if [ ! -z "$PIDS" ]; then
        echo "🛑 Stopping existing MCP server processes: $PIDS"
        kill $PIDS 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        REMAINING=$(pgrep -f "project-watch-mcp" 2>/dev/null || true)
        if [ ! -z "$REMAINING" ]; then
            echo "💀 Force killing remaining processes: $REMAINING"
            kill -9 $REMAINING 2>/dev/null || true
        fi
    else
        echo "ℹ️  No existing MCP server processes found"
    fi
    
    echo ""
    echo "🚀 Starting MCP server with current repository..."
    echo "   Command: uvx project-watch-mcp --repository . --transport stdio --verbose"
    echo ""
    echo "💡 To start the server manually:"
    echo "   uvx project-watch-mcp --repository . --transport stdio"
    echo ""
    echo "🔧 For HTTP mode:"
    echo "   uvx project-watch-mcp --repository . --transport http --port 8080"
    
    # Optionally auto-start the server (commented out by default)
    # echo "🟢 Auto-starting MCP server..."
    # uvx project-watch-mcp --repository . --transport stdio --verbose &
    # echo "📊 MCP server started in background with PID $!"
    
else
    echo "❌ Installation verification failed"
    exit 1
fi