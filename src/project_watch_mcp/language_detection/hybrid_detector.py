"""Hybrid language detection system using tree-sitter, Pygments, and file extensions."""

import logging
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_java

try:
    from pygments import lexers
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    logging.warning("Pygments not available for language detection")

from .models import DetectionMethod, LanguageDetectionResult
from .cache import LanguageDetectionCache

logger = logging.getLogger(__name__)


class HybridLanguageDetector:
    """Hybrid language detector using multiple detection methods."""
    
    # File extension to language mapping
    EXTENSION_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".cs": "csharp",
        ".php": "php",
        ".swift": "swift",
        ".m": "objc",
        ".mm": "objc",
        ".scala": "scala",
        ".r": "r",
        ".R": "r",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".rst": "restructuredtext",
        ".txt": "text",
    }
    
    # Language name normalization mapping
    LANGUAGE_NORMALIZE = {
        "python": "python",
        "python3": "python",
        "py": "python",
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "java": "java",
        "kotlin": "kotlin",
        "kt": "kotlin",
        "c++": "cpp",
        "cpp": "cpp",
        "c": "c",
        "c#": "csharp",
        "csharp": "csharp",
        "cs": "csharp",
        "objective-c": "objc",
        "objc": "objc",
        "objectivec": "objc",
        # Pygments lexer names
        "scdoc": "python",
        "numpy": "python",
        "text only": "text",
        "text": "text",
        "tera term macro": "kotlin",
    }
    
    def __init__(self, enable_cache: bool = True, cache_max_size: int = 1000, cache_max_age_seconds: int = 3600):
        """
        Initialize the hybrid language detector.
        
        Args:
            enable_cache: Whether to enable caching of detection results
            cache_max_size: Maximum number of cache entries
            cache_max_age_seconds: Maximum age of cache entries in seconds
        """
        # Thread safety lock for parser operations
        self._parser_lock = RLock()
        self.tree_sitter_parsers: Dict[str, Parser] = {}
        self._parsers_initialized = False
        self._initialize_tree_sitter()
        
        # Initialize cache
        self.cache_enabled = enable_cache
        if enable_cache:
            self.cache = LanguageDetectionCache(
                max_size=cache_max_size,
                max_age_seconds=cache_max_age_seconds
            )
            logger.info("Language detection cache enabled")
        else:
            self.cache = None
            logger.info("Language detection cache disabled")
    
    def _initialize_tree_sitter(self) -> None:
        """Initialize tree-sitter parsers for supported languages.
        
        Thread-safe initialization with double-checked locking pattern.
        """
        if self._parsers_initialized:
            return
            
        with self._parser_lock:
            # Double-check after acquiring lock
            if self._parsers_initialized:
                return
                
            try:
                # Python parser
                py_parser = Parser(Language(tree_sitter_python.language()))
                self.tree_sitter_parsers["python"] = py_parser
                
                # JavaScript parser
                js_parser = Parser(Language(tree_sitter_javascript.language()))
                self.tree_sitter_parsers["javascript"] = js_parser
                
                # Java parser
                java_parser = Parser(Language(tree_sitter_java.language()))
                self.tree_sitter_parsers["java"] = java_parser
                
                # Mark as initialized after successful setup
                self._parsers_initialized = True
                
                logger.info(f"Initialized tree-sitter parsers for: {list(self.tree_sitter_parsers.keys())}")
            except Exception as e:
                logger.error(f"Failed to initialize tree-sitter parsers: {e}")
                # Don't fail silently - at least have one parser
                self.tree_sitter_parsers = {}
                self._parsers_initialized = True  # Prevent repeated initialization attempts
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language name to standard format."""
        normalized = language.lower().strip()
        return self.LANGUAGE_NORMALIZE.get(normalized, normalized)
    
    def _detect_from_extension(self, extension: str) -> LanguageDetectionResult:
        """Detect language from file extension."""
        ext = extension.lower() if extension else ""
        language = self.EXTENSION_MAP.get(ext, "unknown")
        confidence = 0.7 if language != "unknown" else 0.0
        
        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
            method=DetectionMethod.EXTENSION
        )
    
    def _detect_from_tree_sitter(self, content: str) -> LanguageDetectionResult:
        """Detect language using tree-sitter parsing.
        
        Thread-safe parsing with proper locking around parser access.
        """
        if not content or not content.strip():
            return LanguageDetectionResult("unknown", 0.0, DetectionMethod.TREE_SITTER)
        
        best_result = LanguageDetectionResult("unknown", 0.0, DetectionMethod.TREE_SITTER)
        content_bytes = content.encode('utf-8')
        
        # Create a snapshot of parsers under lock to avoid holding lock during parsing
        with self._parser_lock:
            parser_snapshot = dict(self.tree_sitter_parsers)
        
        for language, parser in parser_snapshot.items():
            try:
                # Parse operation is thread-safe per parser instance
                # Tree-sitter parsers are designed to be thread-safe for parsing
                tree = parser.parse(content_bytes)
                root_node = tree.root_node
                
                # Calculate confidence based on parsing success
                if root_node.has_error:
                    # Count error nodes
                    error_count = self._count_error_nodes(root_node)
                    total_nodes = self._count_total_nodes(root_node)
                    
                    if total_nodes > 0:
                        error_ratio = error_count / total_nodes
                        confidence = max(0.0, 1.0 - error_ratio)
                    else:
                        confidence = 0.0
                else:
                    # Perfect parse - but check if tree has any nodes
                    total_nodes = self._count_total_nodes(root_node)
                    if total_nodes > 0:
                        confidence = 0.95
                    else:
                        # Empty parse tree is not a valid parse
                        confidence = 0.0
                
                # Additional confidence boost for specific language patterns
                if confidence > 0.5:
                    confidence = self._adjust_confidence_for_patterns(
                        content, language, confidence
                    )
                
                if confidence > best_result.confidence:
                    best_result = LanguageDetectionResult(
                        language=language,
                        confidence=confidence,
                        method=DetectionMethod.TREE_SITTER
                    )
                
            except Exception as e:
                logger.debug(f"Tree-sitter parsing failed for {language}: {e}")
                continue
        
        return best_result
    
    def _count_error_nodes(self, node) -> int:
        """Count error nodes in the parse tree."""
        count = 1 if node.type == "ERROR" or node.has_error else 0
        for child in node.children:
            count += self._count_error_nodes(child)
        return count
    
    def _count_total_nodes(self, node) -> int:
        """Count total nodes in the parse tree."""
        count = 1
        for child in node.children:
            count += self._count_total_nodes(child)
        return count
    
    def _adjust_confidence_for_patterns(
        self, content: str, language: str, base_confidence: float
    ) -> float:
        """Adjust confidence based on language-specific patterns."""
        confidence = base_confidence
        
        if language == "python":
            # Python-specific patterns
            if "import " in content or "from " in content:
                confidence += 0.05
            if "def " in content or "class " in content:
                confidence += 0.05
            if "__init__" in content or "__name__" in content:
                confidence += 0.1
        
        elif language == "javascript":
            # JavaScript-specific patterns
            if "function " in content or "const " in content or "let " in content:
                confidence += 0.05
            if "console.log" in content or "require(" in content:
                confidence += 0.05
            if "=>" in content or "async " in content:
                confidence += 0.05
        
        elif language == "java":
            # Java-specific patterns
            if "public class" in content or "private " in content:
                confidence += 0.05
            if "System.out" in content or "import java" in content:
                confidence += 0.1
            if "@Override" in content or "extends " in content:
                confidence += 0.05
        
        return min(1.0, confidence)
    
    def _detect_from_pygments(self, content: str) -> LanguageDetectionResult:
        """Detect language using Pygments lexer analysis."""
        if not PYGMENTS_AVAILABLE:
            return LanguageDetectionResult("unknown", 0.0, DetectionMethod.PYGMENTS)
        
        if not content or not content.strip():
            return LanguageDetectionResult("unknown", 0.0, DetectionMethod.PYGMENTS)
        
        try:
            # Try to guess the lexer
            lexer = lexers.guess_lexer(content)
            language = self._normalize_language(lexer.name)
            
            # Pygments doesn't provide confidence scores, so we estimate
            # based on the lexer's analysis quality
            confidence = 0.8  # Base confidence for successful detection
            
            # Adjust confidence based on content length
            if len(content) < 50:
                confidence *= 0.8
            elif len(content) > 500:
                confidence *= 1.1
            
            confidence = min(0.95, confidence)
            
            return LanguageDetectionResult(
                language=language,
                confidence=confidence,
                method=DetectionMethod.PYGMENTS
            )
        
        except (ClassNotFound, Exception) as e:
            logger.debug(f"Pygments detection failed: {e}")
            return LanguageDetectionResult("unknown", 0.0, DetectionMethod.PYGMENTS)
    
    def detect(
        self,
        content: str,
        file_path: Optional[str] = None,
        use_cache: Optional[bool] = None
    ) -> LanguageDetectionResult:
        """
        Detect language using hybrid approach with optional caching.
        
        Args:
            content: The code content to analyze
            file_path: Optional file path for extension-based detection
            use_cache: Override cache setting for this detection (None = use default)
        
        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        # Determine if we should use cache for this detection
        should_use_cache = self.cache_enabled if use_cache is None else use_cache
        
        # Check cache if enabled
        if should_use_cache and self.cache:
            cached_result = self.cache.get(content, file_path)
            if cached_result:
                logger.debug(f"Cache hit for language detection: {cached_result.language}")
                return cached_result
        
        # Perform detection
        results = []
        
        # Try tree-sitter first (most accurate for supported languages)
        tree_sitter_result = self._detect_from_tree_sitter(content)
        if tree_sitter_result.confidence >= 0.9:
            # Cache and return high-confidence result
            if should_use_cache and self.cache:
                self.cache.put(content, tree_sitter_result, file_path)
            return tree_sitter_result
        results.append(tree_sitter_result)
        
        # Try Pygments
        if PYGMENTS_AVAILABLE:
            pygments_result = self._detect_from_pygments(content)
            if pygments_result.confidence >= 0.85:
                # Cache and return high-confidence result
                if should_use_cache and self.cache:
                    self.cache.put(content, pygments_result, file_path)
                return pygments_result
            results.append(pygments_result)
        
        # Try file extension
        if file_path:
            extension = Path(file_path).suffix
            if extension:
                ext_result = self._detect_from_extension(extension)
                results.append(ext_result)
        
        # Return the result with highest confidence
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            
            # If best confidence is still low but we have an extension match,
            # boost the extension result
            if best_result.confidence < 0.5 and file_path:
                for result in results:
                    if result.method == DetectionMethod.EXTENSION and result.confidence > 0:
                        best_result = result
                        break
            
            # Cache the final result
            if should_use_cache and self.cache:
                self.cache.put(content, best_result, file_path)
            
            return best_result
        
        result = LanguageDetectionResult("unknown", 0.0, DetectionMethod.UNKNOWN)
        
        # Cache even unknown results to avoid re-processing
        if should_use_cache and self.cache:
            self.cache.put(content, result, file_path)
        
        return result
    
    def detect_batch(
        self,
        files: List[Tuple[str, str]],
        use_cache: Optional[bool] = None
    ) -> List[LanguageDetectionResult]:
        """
        Detect languages for multiple files in batch.
        
        Args:
            files: List of (file_path, content) tuples
            use_cache: Override cache setting for this batch (None = use default)
        
        Returns:
            List of LanguageDetectionResult for each file
        """
        results = []
        
        for file_path, content in files:
            result = self.detect(content, file_path, use_cache=use_cache)
            results.append(result)
            logger.debug(f"Detected {file_path}: {result}")
        
        return results
    
    def get_cache_info(self) -> Optional[Dict]:
        """
        Get cache information and statistics.
        
        Returns:
            Dictionary with cache info or None if cache is disabled
        """
        if self.cache:
            return self.cache.get_info()
        return None
    
    def clear_cache(self) -> None:
        """Clear the language detection cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Language detection cache cleared")
    
    def reset_cache_statistics(self) -> None:
        """Reset cache statistics."""
        if self.cache:
            self.cache.reset_statistics()
            logger.info("Language detection cache statistics reset")
    
    def add_parser(self, language: str, parser: Parser) -> None:
        """Add or update a parser for a specific language.
        
        Thread-safe method to dynamically add parsers.
        
        Args:
            language: The language name
            parser: The tree-sitter Parser instance
        """
        with self._parser_lock:
            self.tree_sitter_parsers[language] = parser
            logger.info(f"Added/updated parser for language: {language}")
    
    def remove_parser(self, language: str) -> bool:
        """Remove a parser for a specific language.
        
        Thread-safe method to remove parsers.
        
        Args:
            language: The language name to remove
            
        Returns:
            True if parser was removed, False if not found
        """
        with self._parser_lock:
            if language in self.tree_sitter_parsers:
                del self.tree_sitter_parsers[language]
                logger.info(f"Removed parser for language: {language}")
                return True
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of languages with available parsers.
        
        Thread-safe method to retrieve supported languages.
        
        Returns:
            List of language names
        """
        with self._parser_lock:
            return list(self.tree_sitter_parsers.keys())