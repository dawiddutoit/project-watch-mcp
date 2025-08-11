# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project named "project-watch" that requires Python 3.11 or higher. The project is currently in early development stage (v0.1.0) with minimal structure.

## Development Setup

### Package Management
This project uses `uv` as the package manager (preferred over pip).

```bash
# Install dependencies
uv sync

# Add new dependencies
uv add <package>

# Run Python scripts with uv
uv run python <script.py>
```

### Virtual Environment
The project uses a `.venv` virtual environment that's already created. Use `uv` commands which automatically handle the virtual environment.

## Project Structure

Currently minimal structure:
- `pyproject.toml` - Project configuration and dependencies
- `.venv/` - Python virtual environment (excluded from version control)
- `.idea/` - JetBrains IDE configuration (excluded from version control)

## Development Guidelines

### Adding New Code
- Create a `src/` directory for source code if implementing new features
- Create a `tests/` directory for test files
- Follow Python naming conventions: snake_case for functions/variables, PascalCase for classes

### Testing
Once tests are added:
```bash
# Run tests with pytest (after adding it as dependency)
uv add --dev pytest pytest-cov
uv run pytest
```

### Code Quality
When code is added, set up linting and formatting:
```bash
# Add development dependencies
uv add --dev ruff black

# Format code
uv run black .

# Lint code
uv run ruff check .
```