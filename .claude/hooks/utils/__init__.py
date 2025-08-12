"""Utility modules for Claude Code hooks."""

from .hook_output import (
    HookAction,
    HookResponse,
    HookOutput,
    read_hook_input,
    safe_hook_execution
)

__all__ = [
    'HookAction',
    'HookResponse', 
    'HookOutput',
    'read_hook_input',
    'safe_hook_execution'
]