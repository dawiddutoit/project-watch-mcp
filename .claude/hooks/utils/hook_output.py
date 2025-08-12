#!/usr/bin/env python3
"""
Claude Code Hook Output Utility Module

This module provides standardized functions and classes for formatting hook outputs
according to the Claude Code hooks specification.

Reference: https://docs.anthropic.com/en/docs/claude-code/hooks#hook-output
"""

import json
import sys
from enum import Enum
from typing import Optional, Dict, Any, Union


class HookAction(Enum):
    """Valid hook actions for decision control."""
    ALLOW = "allow"
    BLOCK = "block"
    ASK = "ask"


class HookResponse:
    """
    Standardized hook response structure.
    
    Attributes:
        action: The action to take (allow, block, ask)
        reason: Optional reason for blocking or asking (required for block/ask)
        context: Optional context to add (for SessionStart, UserPromptSubmit)
    """
    def __init__(self, 
                 action: HookAction = HookAction.ALLOW,
                 reason: Optional[str] = None,
                 context: Optional[str] = None):
        self.action = action
        self.reason = reason
        self.context = context
    
    def validate(self) -> None:
        """Validate the response based on action type."""
        if self.action in (HookAction.BLOCK, HookAction.ASK):
            if not self.reason:
                raise ValueError(f"'{self.action.value}' action requires a 'reason'")
        
        if self.action != HookAction.ALLOW and self.context:
            raise ValueError(f"'{self.action.value}' action should not include 'context'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.validate()
        
        result = {"action": self.action.value}
        
        if self.reason is not None:
            result["reason"] = self.reason
        if self.context is not None:
            result["context"] = self.context
            
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def output(self) -> None:
        """Output the response to stdout (for use in hooks)."""
        print(self.to_json())
        sys.stdout.flush()


class HookOutput:
    """
    Utility class for creating and outputting hook responses.
    """
    
    @staticmethod
    def allow(context: Optional[str] = None) -> HookResponse:
        """
        Create an allow response.
        
        Args:
            context: Optional context to add (for SessionStart, UserPromptSubmit)
            
        Returns:
            HookResponse configured for allow action
        """
        return HookResponse(
            action=HookAction.ALLOW,
            context=context
        )
    
    @staticmethod
    def block(reason: str) -> HookResponse:
        """
        Create a block response.
        
        Args:
            reason: Required reason for blocking
            
        Returns:
            HookResponse configured for block action
        """
        return HookResponse(
            action=HookAction.BLOCK,
            reason=reason
        )
    
    @staticmethod
    def ask(reason: str) -> HookResponse:
        """
        Create an ask response to request user permission.
        
        Args:
            reason: Required reason/question for the user
            
        Returns:
            HookResponse configured for ask action
        """
        return HookResponse(
            action=HookAction.ASK,
            reason=reason
        )
    
    @staticmethod
    def error(message: str, allow_continuation: bool = True) -> HookResponse:
        """
        Create an error response.
        
        By default, errors allow continuation to avoid blocking the session.
        
        Args:
            message: Error message
            allow_continuation: Whether to allow operation despite error
            
        Returns:
            HookResponse with error information
        """
        if allow_continuation:
            return HookResponse(
                action=HookAction.ALLOW,
                context=f"⚠️ Hook Error: {message}\n\nOperation will continue."
            )
        else:
            return HookResponse(
                action=HookAction.BLOCK,
                reason=f"Hook Error: {message}"
            )


def read_hook_input() -> Dict[str, Any]:
    """
    Read and parse hook input from stdin.
    
    Returns:
        Parsed JSON input as dictionary
        
    Raises:
        json.JSONDecodeError: If input is not valid JSON
    """
    return json.loads(sys.stdin.read())


def safe_hook_execution(func):
    """
    Decorator for safe hook execution with error handling.
    
    This decorator ensures that hooks always return a valid response,
    even if an error occurs during execution.
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, HookResponse):
                result.output()
            elif result is not None:
                print(json.dumps(result))
                sys.stdout.flush()
            else:
                HookOutput.allow().output()
        except Exception as e:
            import traceback
            error_msg = f"Hook error: {str(e)}\n{traceback.format_exc()}"
            HookOutput.error(error_msg, allow_continuation=True).output()
    
    return wrapper