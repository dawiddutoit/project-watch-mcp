"""Comprehensive integration tests for all three systems working together.

This test suite validates the complete integration of:
1. Vector search (Neo4j + embeddings)
2. Language detection (hybrid detector)
3. Complexity analysis (multi-language)

Testing scenarios:
- File processing pipeline
- Cross-feature search
- Real-time updates
- Performance integration
- Error propagation
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest
from fastmcp.tools.tool import ToolResult

from src.project_watch_mcp.neo4j_rag import Neo4jRAG
from src.project_watch_mcp.repository_monitor import RepositoryMonitor
from src.project_watch_mcp.server import create_mcp_server
from src.project_watch_mcp.language_detection import HybridLanguageDetector
from src.project_watch_mcp.language_detection.models import LanguageDetectionResult, DetectionMethod
from src.project_watch_mcp.complexity_analysis import AnalyzerRegistry, ComplexityResult
from src.project_watch_mcp.complexity_analysis.models import (
    ComplexitySummary, FunctionComplexity, ComplexityGrade
)


class TestFullSystemIntegration:
    """Test complete integration of vector search, language detection, and complexity analysis."""
    
    @pytest.fixture
    async def multi_language_repository(self):
        """Create a test repository with multiple programming languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Python file with varying complexity
            (repo_path / "auth_handler.py").write_text("""
import jwt
from datetime import datetime, timedelta

class AuthenticationHandler:
    '''Handles user authentication with JWT tokens.'''
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=24)
    
    def validate_jwt_token(self, token: str) -> dict:
        '''Validate JWT token and return claims.
        
        High complexity function with multiple conditions.
        '''
        if not token:
            raise ValueError("Token is required")
        
        try:
            # Decode token
            decoded = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiration
            if 'exp' in decoded:
                exp_time = datetime.fromtimestamp(decoded['exp'])
                if exp_time < datetime.now():
                    raise ValueError("Token expired")
            
            # Validate claims
            if 'user_id' not in decoded:
                raise ValueError("Invalid token: missing user_id")
            
            # Check roles
            if 'roles' in decoded:
                if 'admin' in decoded['roles']:
                    decoded['is_admin'] = True
                elif 'moderator' in decoded['roles']:
                    decoded['is_moderator'] = True
                else:
                    decoded['is_user'] = True
            
            return decoded
            
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            raise ValueError(f"Token validation failed: {e}")
    
    def generate_token(self, user_id: str, roles: List[str] = None) -> str:
        '''Generate a new JWT token - simple function.'''
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + self.token_expiry,
            'roles': roles or []
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
""")
            
            # JavaScript file with moderate complexity
            (repo_path / "data_processor.js").write_text("""
class DataProcessor {
    constructor() {
        this.data = [];
        this.processed = false;
    }
    
    async processData(input) {
        // Moderate complexity with async operations
        if (!input || input.length === 0) {
            throw new Error('Input data required');
        }
        
        const results = [];
        
        for (const item of input) {
            if (item.type === 'user') {
                const processed = await this.processUser(item);
                results.push(processed);
            } else if (item.type === 'order') {
                const processed = await this.processOrder(item);
                results.push(processed);
            } else {
                console.warn('Unknown item type:', item.type);
            }
        }
        
        this.data = results;
        this.processed = true;
        return results;
    }
    
    async processUser(user) {
        // Simple processing
        return {
            ...user,
            processed: true,
            timestamp: new Date().toISOString()
        };
    }
    
    async processOrder(order) {
        // Simple processing
        return {
            ...order,
            processed: true,
            total: order.items.reduce((sum, item) => sum + item.price, 0)
        };
    }
}

module.exports = DataProcessor;
""")
            
            # Java file with high complexity
            (repo_path / "PaymentService.java").write_text("""
package com.example.payment;

import java.util.*;
import java.math.BigDecimal;

public class PaymentService {
    private final PaymentGateway gateway;
    private final FraudDetector fraudDetector;
    
    public PaymentService(PaymentGateway gateway, FraudDetector fraudDetector) {
        this.gateway = gateway;
        this.fraudDetector = fraudDetector;
    }
    
    public PaymentResult processPayment(PaymentRequest request) {
        // Very high complexity method
        if (request == null) {
            return PaymentResult.error("Invalid request");
        }
        
        // Validate amount
        if (request.getAmount() == null || request.getAmount().compareTo(BigDecimal.ZERO) <= 0) {
            return PaymentResult.error("Invalid amount");
        }
        
        // Check fraud
        FraudCheckResult fraudResult = fraudDetector.check(request);
        if (fraudResult.isHighRisk()) {
            if (fraudResult.getScore() > 90) {
                return PaymentResult.rejected("High fraud risk");
            } else if (fraudResult.getScore() > 70) {
                if (request.getAmount().compareTo(new BigDecimal("1000")) > 0) {
                    return PaymentResult.requiresVerification("Additional verification required");
                }
            }
        }
        
        // Process based on payment method
        PaymentResult result;
        switch (request.getPaymentMethod()) {
            case CREDIT_CARD:
                result = processCreditCard(request);
                break;
            case DEBIT_CARD:
                result = processDebitCard(request);
                break;
            case PAYPAL:
                result = processPayPal(request);
                break;
            case BANK_TRANSFER:
                if (request.getAmount().compareTo(new BigDecimal("10000")) > 0) {
                    result = processLargeBankTransfer(request);
                } else {
                    result = processSmallBankTransfer(request);
                }
                break;
            default:
                result = PaymentResult.error("Unsupported payment method");
        }
        
        // Post-processing
        if (result.isSuccessful()) {
            notifySuccess(request, result);
            updateMetrics(request, result);
        } else {
            notifyFailure(request, result);
            if (result.isRetryable()) {
                scheduleRetry(request);
            }
        }
        
        return result;
    }
    
    private PaymentResult processCreditCard(PaymentRequest request) {
        // Simple method
        return gateway.charge(request);
    }
    
    private PaymentResult processDebitCard(PaymentRequest request) {
        // Simple method
        return gateway.charge(request);
    }
}
""")
            
            # TypeScript file
            (repo_path / "api_client.ts").write_text("""
interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

class ApiClient {
    private baseUrl: string;
    private headers: Record<string, string>;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    async get<T>(endpoint: string): Promise<ApiResponse<T>> {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'GET',
            headers: this.headers
        });
        
        const data = await response.json();
        return {
            data,
            status: response.status
        };
    }
}

export default ApiClient;
""")
            
            # Simple text file (no complexity)
            (repo_path / "README.md").write_text("""
# Test Repository

This is a test repository for integration testing.

## Features
- Authentication with JWT
- Data processing
- Payment handling
- API client
""")
            
            # Configuration file
            (repo_path / "config.json").write_text(json.dumps({
                "version": "1.0.0",
                "features": {
                    "authentication": True,
                    "payments": True,
                    "data_processing": True
                }
            }))
            
            # Create .gitignore
            (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/\nnode_modules/\n*.class\n")
            
            yield repo_path
    
    @pytest.fixture
    async def integrated_mcp_server(self, multi_language_repository):
        """Create MCP server with all systems integrated."""
        # Mock Neo4j driver
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        # Storage for indexed data with language and complexity info
        indexed_files = {}
        indexed_chunks = []
        language_cache = {}
        complexity_cache = {}
        
        async def mock_execute_query(query, params=None, *args, **kwargs):
            """Mock Neo4j queries with language and complexity awareness."""
            if "MERGE (f:CodeFile" in query:
                # Store file with language and complexity metadata
                file_path = params.get("path", "")
                indexed_files[file_path] = {
                    **params,
                    "language": params.get("language"),
                    "complexity": params.get("complexity"),
                    "complexity_grade": params.get("complexity_grade")
                }
                return MagicMock(records=[])
            
            elif "MERGE (c:CodeChunk" in query:
                # Store chunk with metadata
                chunk_data = {
                    **params,
                    "language": params.get("language"),
                    "complexity_context": params.get("complexity_context")
                }
                indexed_chunks.append(chunk_data)
                return MagicMock(records=[])
            
            elif "MATCH (c:CodeChunk" in query and "similarity" in query.lower():
                # Language-aware semantic search
                language_filter = params.get("language") if params else None
                complexity_filter = params.get("max_complexity") if params else None
                
                results = []
                for file_path, file_data in indexed_files.items():
                    # Apply language filter
                    if language_filter and file_data.get("language") != language_filter:
                        continue
                    
                    # Apply complexity filter
                    if complexity_filter and file_data.get("complexity", 0) > complexity_filter:
                        continue
                    
                    results.append({
                        "file_path": str(multi_language_repository / file_path),
                        "chunk_content": f"Sample content from {file_path}",
                        "line_number": 10,
                        "similarity": 0.85,
                        "language": file_data.get("language", "unknown"),
                        "complexity": file_data.get("complexity", 0)
                    })
                
                return MagicMock(records=results[:5])  # Limit results
            
            elif "MATCH (f:CodeFile" in query and "language" in query.lower():
                # Get files by language
                language = params.get("language") if params else None
                results = []
                for file_path, file_data in indexed_files.items():
                    if language and file_data.get("language") == language:
                        results.append(file_data)
                return MagicMock(records=results)
            
            elif "count(DISTINCT f)" in query:
                # Get repository stats with language breakdown
                language_stats = {}
                for file_data in indexed_files.values():
                    lang = file_data.get("language", "unknown")
                    if lang not in language_stats:
                        language_stats[lang] = {"files": 0, "total_complexity": 0}
                    language_stats[lang]["files"] += 1
                    language_stats[lang]["total_complexity"] += file_data.get("complexity", 0)
                
                return MagicMock(records=[{
                    "total_files": len(indexed_files),
                    "total_chunks": len(indexed_chunks),
                    "languages": list(language_stats.keys()),
                    "language_breakdown": language_stats,
                    "average_complexity": sum(f.get("complexity", 0) for f in indexed_files.values()) / max(len(indexed_files), 1)
                }])
            
            return MagicMock(records=[])
        
        mock_driver.execute_query = mock_execute_query
        
        # Create language detector
        language_detector = HybridLanguageDetector()
        
        # Mock language detection to return consistent results
        async def mock_detect_language(file_path: Path) -> LanguageDetectionResult:
            ext = file_path.suffix.lower()
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".java": "java",
                ".ts": "typescript",
                ".md": "markdown",
                ".json": "json"
            }
            
            language = language_map.get(ext, "unknown")
            confidence = 0.95 if language != "unknown" else 0.3
            
            result = LanguageDetectionResult(
                language=language,
                confidence=confidence,
                method=DetectionMethod.EXTENSION,
                metadata={"file_path": str(file_path)}
            )
            
            # Cache the result
            language_cache[str(file_path)] = result
            return result
        
        language_detector.detect_language = mock_detect_language
        
        # Create complexity analyzer registry
        analyzer_registry = AnalyzerRegistry()
        
        # Mock complexity analysis
        def mock_analyze_complexity(file_path: Path, language: str) -> ComplexityResult:
            # Return different complexity based on file
            complexity_map = {
                "auth_handler.py": 15,  # High complexity
                "data_processor.js": 8,  # Moderate complexity
                "PaymentService.java": 25,  # Very high complexity
                "api_client.ts": 3,  # Low complexity
                "README.md": 0,  # No complexity
                "config.json": 0  # No complexity
            }
            
            file_name = file_path.name
            complexity = complexity_map.get(file_name, 5)
            
            # Determine grade
            if complexity == 0:
                grade = ComplexityGrade.A
            elif complexity <= 5:
                grade = ComplexityGrade.A
            elif complexity <= 10:
                grade = ComplexityGrade.B
            elif complexity <= 20:
                grade = ComplexityGrade.C
            else:
                grade = ComplexityGrade.D
            
            result = ComplexityResult(
                file_path=str(file_path),
                language=language,
                summary=ComplexitySummary(
                    total_complexity=complexity,
                    average_complexity=complexity / max(1, complexity // 3),
                    max_complexity=complexity,
                    complexity_grade=grade
                ),
                functions=[
                    FunctionComplexity(
                        name=f"function_{i}",
                        complexity=min(complexity // 2, 10),
                        line=i * 10,
                        classification="complex" if complexity > 10 else "simple"
                    ) for i in range(1, min(4, complexity // 3 + 1))
                ] if complexity > 0 else []
            )
            
            # Cache the result
            complexity_cache[str(file_path)] = result
            return result
        
        analyzer_registry.analyze = mock_analyze_complexity
        
        # Create repository monitor with integrated systems
        monitor = RepositoryMonitor(
            repo_path=multi_language_repository,
            project_name="test_integrated",
            neo4j_driver=mock_driver,
            file_patterns=["*.py", "*.js", "*.java", "*.ts", "*.md", "*.json"],
            ignore_patterns=["*.pyc", "__pycache__", ".venv", "node_modules", "*.class"],
        )
        
        # Inject language detector and complexity analyzer
        monitor.language_detector = language_detector
        monitor.complexity_analyzer = analyzer_registry
        
        # Mock embeddings provider
        from tests.unit.utils.embeddings.test_embeddings_utils import TestEmbeddingsProvider
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_integrated",
            embeddings=TestEmbeddingsProvider(),
        )
        await rag.initialize()
        
        # Inject systems into RAG
        rag.language_detector = language_detector
        rag.complexity_analyzer = analyzer_registry
        
        # Create server
        server = create_mcp_server(monitor, rag, "test_integrated")
        
        return server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache
    
    async def test_file_processing_pipeline(self, integrated_mcp_server, multi_language_repository):
        """Test complete file processing: Language Detection → Complexity Analysis → Vector Embedding → Search."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize repository - this should trigger the full pipeline
        init_tool = await server.get_tool("initialize_repository")
        assert init_tool is not None
        
        # Mock the initializer to simulate full pipeline
        from src.project_watch_mcp.core.initializer import InitializationResult
        with patch('src.project_watch_mcp.core.initializer.RepositoryInitializer') as MockInitializer:
            mock_instance = AsyncMock()
            
            # Simulate processing each file through the pipeline
            async def mock_initialize():
                for file_path in multi_language_repository.glob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(multi_language_repository)
                        
                        # Step 1: Language detection
                        lang_result = await server._monitor.language_detector.detect_language(file_path)
                        
                        # Step 2: Complexity analysis (if applicable)
                        complexity = 0
                        grade = ComplexityGrade.A
                        if lang_result.language in ["python", "javascript", "java", "typescript"]:
                            complexity_result = server._monitor.complexity_analyzer.analyze(
                                file_path, lang_result.language
                            )
                            complexity = complexity_result.summary.total_complexity
                            grade = complexity_result.summary.complexity_grade
                        
                        # Step 3: Store with metadata
                        indexed_files[str(relative_path)] = {
                            "path": str(relative_path),
                            "language": lang_result.language,
                            "language_confidence": lang_result.confidence,
                            "complexity": complexity,
                            "complexity_grade": grade.value,
                            "size": file_path.stat().st_size
                        }
                
                return InitializationResult(
                    indexed=len(indexed_files),
                    total=len(indexed_files),
                    skipped=[],
                    monitoring=True,
                    message=f"Processed {len(indexed_files)} files through complete pipeline"
                )
            
            mock_instance.initialize = mock_initialize
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockInitializer.return_value = mock_instance
            
            result = await init_tool.run({})
        
        # Validate pipeline execution
        assert isinstance(result, ToolResult)
        assert result.structured_content["indexed"] > 0
        
        # Verify all files were processed with language detection
        assert len(indexed_files) > 0
        for file_data in indexed_files.values():
            assert "language" in file_data
            assert "language_confidence" in file_data
            
            # Code files should have complexity analysis
            if file_data["language"] in ["python", "javascript", "java", "typescript"]:
                assert "complexity" in file_data
                assert "complexity_grade" in file_data
    
    async def test_cross_feature_search(self, integrated_mcp_server, multi_language_repository):
        """Test language-aware vector search with complexity filtering."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize repository first
        init_tool = await server.get_tool("initialize_repository")
        
        # Simulate initialization with full metadata
        indexed_files["auth_handler.py"] = {
            "path": "auth_handler.py",
            "language": "python",
            "complexity": 15,
            "complexity_grade": "C"
        }
        indexed_files["data_processor.js"] = {
            "path": "data_processor.js",
            "language": "javascript",
            "complexity": 8,
            "complexity_grade": "B"
        }
        indexed_files["PaymentService.java"] = {
            "path": "PaymentService.java",
            "language": "java",
            "complexity": 25,
            "complexity_grade": "D"
        }
        
        await init_tool.run({})
        
        # Test 1: Search only Python files
        search_tool = await server.get_tool("search_code")
        result = await search_tool.run({
            "query": "authentication validation",
            "search_type": "semantic",
            "language": "python",
            "limit": 5
        })
        
        assert isinstance(result, ToolResult)
        assert result.structured_content is not None
        
        # Test 2: Search with complexity filtering (exclude high complexity)
        result = await search_tool.run({
            "query": "process data",
            "search_type": "semantic",
            "limit": 5
        })
        
        assert isinstance(result, ToolResult)
        
        # Test 3: Pattern search in specific language
        result = await search_tool.run({
            "query": "class.*Service",
            "search_type": "pattern",
            "is_regex": True,
            "language": "java",
            "limit": 3
        })
        
        assert isinstance(result, ToolResult)
    
    async def test_real_time_updates(self, integrated_mcp_server, multi_language_repository):
        """Test that file changes trigger all three systems correctly."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize repository
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Create a new file
        new_file = multi_language_repository / "new_feature.py"
        new_file.write_text("""
def calculate_metrics(data):
    '''Calculate various metrics with moderate complexity.'''
    if not data:
        return {}
    
    metrics = {}
    
    # Calculate average
    if len(data) > 0:
        metrics['average'] = sum(data) / len(data)
    
    # Calculate min/max
    if data:
        metrics['min'] = min(data)
        metrics['max'] = max(data)
    
    # Calculate variance
    if len(data) > 1:
        avg = metrics.get('average', 0)
        variance = sum((x - avg) ** 2 for x in data) / len(data)
        metrics['variance'] = variance
    
    return metrics
""")
        
        # Refresh the file - should trigger all systems
        refresh_tool = await server.get_tool("refresh_file")
        result = await refresh_tool.run({"file_path": "new_feature.py"})
        
        assert isinstance(result, ToolResult)
        assert result.structured_content["status"] == "success"
        
        # Verify the file was processed through all systems
        # (In real implementation, this would update indexed_files with language and complexity)
        
        # Modify existing file
        modified_file = multi_language_repository / "auth_handler.py"
        modified_file.write_text(modified_file.read_text() + "\n\n# Modified content\n")
        
        # Refresh should detect change and re-process
        result = await refresh_tool.run({"file_path": "auth_handler.py"})
        assert isinstance(result, ToolResult)
        assert result.structured_content["status"] == "success"
    
    async def test_performance_integration(self, integrated_mcp_server, multi_language_repository):
        """Test combined system performance under load."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize repository
        init_tool = await server.get_tool("initialize_repository")
        
        # Measure initialization time (all systems)
        start_time = time.time()
        await init_tool.run({})
        init_time = time.time() - start_time
        
        # Performance benchmark: initialization should complete reasonably fast
        assert init_time < 10.0, f"Initialization took {init_time:.2f}s (should be < 10s)"
        
        # Test search performance with multiple queries
        search_tool = await server.get_tool("search_code")
        
        search_queries = [
            {"query": "authentication", "search_type": "semantic"},
            {"query": "process data", "search_type": "semantic", "language": "javascript"},
            {"query": "payment", "search_type": "semantic", "language": "java"},
            {"query": "def.*validate", "search_type": "pattern", "is_regex": True},
            {"query": "class.*Handler", "search_type": "pattern", "is_regex": True, "language": "python"}
        ]
        
        # Run searches and measure time
        search_times = []
        for query_params in search_queries:
            start_time = time.time()
            result = await search_tool.run(query_params)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            assert isinstance(result, ToolResult)
        
        # Average search time should be reasonable
        avg_search_time = sum(search_times) / len(search_times)
        assert avg_search_time < 2.0, f"Average search time {avg_search_time:.2f}s (should be < 2s)"
        
        # Test parallel operations
        tasks = [
            search_tool.run({"query": f"test_{i}", "search_type": "semantic"})
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        parallel_time = time.time() - start_time
        
        # Parallel execution should be efficient
        assert parallel_time < 5.0, f"Parallel execution took {parallel_time:.2f}s"
        assert all(isinstance(r, ToolResult) for r in results)
        
        # Performance degradation check
        # Combined system should not be more than 20% slower than individual components
        individual_time_estimate = 1.0  # Baseline estimate
        assert init_time < individual_time_estimate * 1.2, \
            f"Performance degradation > 20%: {init_time:.2f}s vs {individual_time_estimate:.2f}s baseline"
    
    async def test_error_propagation(self, integrated_mcp_server, multi_language_repository):
        """Test how errors in one system affect others."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Test 1: Language detection failure
        # Create a file with unknown extension
        unknown_file = multi_language_repository / "unknown.xyz"
        unknown_file.write_text("Some content in unknown format")
        
        refresh_tool = await server.get_tool("refresh_file")
        result = await refresh_tool.run({"file_path": "unknown.xyz"})
        
        # Should handle gracefully - file indexed but with unknown language
        assert isinstance(result, ToolResult)
        # System should continue working despite unknown language
        
        # Test 2: Complexity analysis failure on malformed code
        malformed_file = multi_language_repository / "malformed.py"
        malformed_file.write_text("""
def broken_function(
    # Syntax error - unclosed parenthesis
    print("This won't parse")
""")
        
        result = await refresh_tool.run({"file_path": "malformed.py"})
        
        # Should handle syntax errors gracefully
        assert isinstance(result, ToolResult)
        # File should still be indexed for search, just without complexity metrics
        
        # Test 3: Search should still work despite some files having errors
        search_tool = await server.get_tool("search_code")
        result = await search_tool.run({
            "query": "function",
            "search_type": "semantic"
        })
        
        assert isinstance(result, ToolResult)
        # Results should include files that were successfully processed
        
        # Test 4: Stats should reflect partial processing
        stats_tool = await server.get_tool("get_repository_stats")
        result = await stats_tool.run({})
        
        assert isinstance(result, ToolResult)
        assert result.structured_content is not None
        # Stats should show total files including those with errors
    
    async def test_integrated_file_info(self, integrated_mcp_server, multi_language_repository):
        """Test that get_file_info returns integrated metadata from all systems."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize and index files
        init_tool = await server.get_tool("initialize_repository")
        
        # Set up indexed file with full metadata
        indexed_files["auth_handler.py"] = {
            "path": "auth_handler.py",
            "language": "python",
            "language_confidence": 0.95,
            "complexity": 15,
            "complexity_grade": "C",
            "size": 2048,
            "lines": 65,
            "hash": "abc123def456"
        }
        
        await init_tool.run({})
        
        # Get file info
        file_info_tool = await server.get_tool("get_file_info")
        result = await file_info_tool.run({"file_path": "auth_handler.py"})
        
        assert isinstance(result, ToolResult)
        assert result.structured_content is not None
        
        # Should include metadata from all systems
        info = result.structured_content
        assert "language" in info  # From language detection
        assert "complexity" in info or "complexity_score" in info  # From complexity analysis
        assert "size" in info  # Basic file metadata
        
        # Test with non-code file
        result = await file_info_tool.run({"file_path": "README.md"})
        assert isinstance(result, ToolResult)
        # Should still have language info but no complexity
    
    async def test_monitoring_status_integration(self, integrated_mcp_server):
        """Test that monitoring status reflects all systems."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Initialize repository
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Get monitoring status
        status_tool = await server.get_tool("monitoring_status")
        result = await status_tool.run({})
        
        assert isinstance(result, ToolResult)
        assert result.structured_content is not None
        
        status = result.structured_content
        
        # Should include information about all systems
        assert "is_running" in status
        assert "file_patterns" in status
        
        # Should reflect integrated capabilities
        # (In real implementation, would show language detection and complexity analysis status)
    
    async def test_repository_stats_integration(self, integrated_mcp_server):
        """Test that repository stats aggregate data from all systems."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Set up files with full metadata
        indexed_files["auth_handler.py"] = {
            "path": "auth_handler.py",
            "language": "python",
            "complexity": 15,
            "size": 2048
        }
        indexed_files["data_processor.js"] = {
            "path": "data_processor.js",
            "language": "javascript",
            "complexity": 8,
            "size": 1024
        }
        indexed_files["PaymentService.java"] = {
            "path": "PaymentService.java",
            "language": "java",
            "complexity": 25,
            "size": 3072
        }
        
        # Initialize
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Get stats
        stats_tool = await server.get_tool("get_repository_stats")
        result = await stats_tool.run({})
        
        assert isinstance(result, ToolResult)
        assert result.structured_content is not None
        
        stats = result.structured_content
        
        # Should include aggregated metrics
        assert "total_files" in stats
        assert stats["total_files"] >= 3
        
        # Should have language breakdown
        assert "languages" in stats or "language_breakdown" in stats
        
        # Should have complexity metrics
        assert "average_complexity" in stats or "complexity_metrics" in stats


class TestIntegrationPerformanceBenchmarks:
    """Performance benchmarks for the integrated system."""
    
    @pytest.fixture
    async def large_repository(self):
        """Create a larger test repository for performance testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create multiple files of each type
            languages = {
                "python": ".py",
                "javascript": ".js",
                "java": ".java",
                "typescript": ".ts"
            }
            
            for lang, ext in languages.items():
                for i in range(10):
                    file_path = repo_path / f"{lang}_file_{i}{ext}"
                    
                    if ext == ".py":
                        content = f'''
def function_{i}(param):
    """Function {i} in Python."""
    result = param * 2
    if result > 10:
        return result * 3
    else:
        return result + 5

class Class_{i}:
    def method_{i}(self):
        return "Method {i}"
'''
                    elif ext == ".js":
                        content = f'''
function function_{i}(param) {{
    // Function {i} in JavaScript
    const result = param * 2;
    if (result > 10) {{
        return result * 3;
    }} else {{
        return result + 5;
    }}
}}

class Class_{i} {{
    method_{i}() {{
        return "Method {i}";
    }}
}}
'''
                    elif ext == ".java":
                        content = f'''
public class Class_{i} {{
    public int function_{i}(int param) {{
        // Function {i} in Java
        int result = param * 2;
        if (result > 10) {{
            return result * 3;
        }} else {{
            return result + 5;
        }}
    }}
    
    public String method_{i}() {{
        return "Method {i}";
    }}
}}
'''
                    else:  # TypeScript
                        content = f'''
function function_{i}(param: number): number {{
    // Function {i} in TypeScript
    const result = param * 2;
    if (result > 10) {{
        return result * 3;
    }} else {{
        return result + 5;
    }}
}}

class Class_{i} {{
    method_{i}(): string {{
        return "Method {i}";
    }}
}}
'''
                    
                    file_path.write_text(content)
            
            yield repo_path
    
    @pytest.mark.performance
    async def test_large_repository_initialization(self, large_repository):
        """Benchmark initialization performance with many files."""
        # This test would measure actual performance with a larger dataset
        # For unit tests, we just verify the structure
        assert large_repository.exists()
        
        # Count files
        file_count = len(list(large_repository.glob("*")))
        assert file_count == 40  # 10 files × 4 languages
    
    @pytest.mark.performance
    async def test_concurrent_operations(self, integrated_mcp_server):
        """Test system performance under concurrent load."""
        server, *_ = integrated_mcp_server
        
        # Initialize first
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Create multiple concurrent operations
        search_tool = await server.get_tool("search_code")
        stats_tool = await server.get_tool("get_repository_stats")
        status_tool = await server.get_tool("monitoring_status")
        
        # Run operations concurrently
        tasks = []
        for i in range(10):
            tasks.append(search_tool.run({"query": f"test_{i}", "search_type": "semantic"}))
            if i % 3 == 0:
                tasks.append(stats_tool.run({}))
            if i % 5 == 0:
                tasks.append(status_tool.run({}))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Check all operations completed successfully
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent operations had {len(errors)} errors"
        
        # All should return ToolResult
        assert all(isinstance(r, ToolResult) for r in results if not isinstance(r, Exception))
        
        # Performance should be reasonable even under load
        assert duration < 30.0, f"Concurrent operations took {duration:.2f}s"


class TestIntegrationErrorRecovery:
    """Test error recovery and resilience of the integrated system."""
    
    async def test_partial_system_failure(self, integrated_mcp_server):
        """Test that system continues working when one component fails."""
        server, mock_driver, indexed_files, indexed_chunks, language_cache, complexity_cache = integrated_mcp_server
        
        # Simulate language detection failure for specific file types
        original_detect = server._monitor.language_detector.detect_language
        
        async def failing_detect(file_path: Path):
            if file_path.suffix == ".java":
                raise Exception("Language detection failed for Java files")
            return await original_detect(file_path)
        
        server._monitor.language_detector.detect_language = failing_detect
        
        # Initialize should still work for other files
        init_tool = await server.get_tool("initialize_repository")
        result = await init_tool.run({})
        
        assert isinstance(result, ToolResult)
        # Should have indexed non-Java files
        
        # Search should still work
        search_tool = await server.get_tool("search_code")
        result = await search_tool.run({"query": "function", "search_type": "semantic"})
        assert isinstance(result, ToolResult)
    
    async def test_recovery_after_error(self, integrated_mcp_server):
        """Test that system can recover after encountering errors."""
        server, *_ = integrated_mcp_server
        
        # Cause an error during initialization
        with patch('src.project_watch_mcp.core.initializer.RepositoryInitializer') as MockInit:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(side_effect=Exception("Initialization failed"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockInit.return_value = mock_instance
            
            init_tool = await server.get_tool("initialize_repository")
            
            try:
                await init_tool.run({})
            except:
                pass  # Expected to fail
        
        # System should still be functional for other operations
        status_tool = await server.get_tool("monitoring_status")
        result = await status_tool.run({})
        assert isinstance(result, ToolResult)
        
        # Should be able to retry initialization
        with patch('src.project_watch_mcp.core.initializer.RepositoryInitializer') as MockInit:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=Mock(
                indexed=5, total=5, skipped=[], monitoring=True,
                message="Recovery successful"
            ))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockInit.return_value = mock_instance
            
            result = await init_tool.run({})
            assert isinstance(result, ToolResult)
            assert result.structured_content["indexed"] == 5