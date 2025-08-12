#!/usr/bin/env python3
"""Session-start hook that reminds Claude to read CLAUDE.md."""

import json
import sys


def main():
    """Main entry point for session-start hook."""
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "Read and follow the instructions in CLAUDE.md"
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())