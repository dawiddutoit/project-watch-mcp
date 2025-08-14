"""Data models for language detection."""

from dataclasses import dataclass
from enum import Enum


class DetectionMethod(Enum):
    """Methods used for language detection."""
    
    TREE_SITTER = "tree_sitter"
    PYGMENTS = "pygments"
    EXTENSION = "extension"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Result of language detection with confidence score."""
    
    language: str
    confidence: float
    method: DetectionMethod
    
    def __repr__(self) -> str:
        return f"LanguageDetectionResult(language='{self.language}', confidence={self.confidence:.2f}, method={self.method.value})"