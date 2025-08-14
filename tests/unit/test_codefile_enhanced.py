"""Test suite for enhanced CodeFile functionality."""

from datetime import datetime
from pathlib import Path

from src.project_watch_mcp.neo4j_rag import CodeFile


class TestCodeFileEnhanced:
    """Test suite for enhanced CodeFile features."""

    def test_basic_initialization(self):
        """Test basic CodeFile initialization with required fields."""
        code_file = CodeFile(
            project_name="test_project",
            path=Path("/project/src/main.py"),
            content="def main():\n    pass",
            language="python",
            size=22,
            last_modified=datetime.now(),
        )

        assert code_file.project_name == "test_project"
        assert code_file.filename == "main.py"
        assert code_file.line_count == 2
        assert code_file.file_category == "source"

    def test_filename_auto_population(self):
        """Test that filename is auto-populated from path."""
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/src/utils/helper.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
        )

        assert code_file.filename == "helper.py"

    def test_explicit_filename(self):
        """Test that explicit filename is preserved."""
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/src/main.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
            filename="custom_name.py",
        )

        assert code_file.filename == "custom_name.py"

    def test_test_file_detection(self):
        """Test detection of test files."""
        test_cases = [
            (Path("/project/test_main.py"), True),
            (Path("/project/main_test.py"), True),
            (Path("/project/main.spec.js"), True),
            (Path("/project/main.test.js"), True),
            (Path("/project/tests/main.py"), True),
            (Path("/project/__tests__/main.js"), True),
            (Path("/project/src/main.py"), False),
        ]

        for path, expected_is_test in test_cases:
            code_file = CodeFile(
                project_name="test",
                path=path,
                content="",
                language="python",
                size=0,
                last_modified=datetime.now(),
            )
            assert code_file.is_test == expected_is_test, f"Failed for {path}"

    def test_config_file_detection(self):
        """Test detection of configuration files."""
        config_cases = [
            (Path("/project/config.py"), True),
            (Path("/project/settings.py"), True),
            (Path("/project/app.yaml"), True),
            (Path("/project/config.yml"), True),
            (Path("/project/setup.toml"), True),
            (Path("/project/.env"), True),
            (Path("/project/data.json"), True),
            (Path("/project/package.json"), True),
            (Path("/project/pyproject.toml"), True),
            (Path("/project/tsconfig.json"), True),
            (Path("/project/webpack.config.js"), True),
            (Path("/project/jest.config.js"), True),
            (Path("/project/.bashrc"), True),
            (Path("/project/main.py"), False),
        ]

        for path, expected_is_config in config_cases:
            code_file = CodeFile(
                project_name="test",
                path=path,
                content="",
                language="python",
                size=0,
                last_modified=datetime.now(),
            )
            assert code_file.is_config == expected_is_config, f"Failed for {path}"

    def test_resource_file_detection(self):
        """Test detection of resource files."""
        resource_cases = [
            (Path("/project/data.sql"), True),
            (Path("/project/config.xml"), True),
            (Path("/project/data.csv"), True),
            (Path("/project/notes.txt"), True),
            (Path("/project/data.dat"), True),
            (Path("/project/logo.png"), True),
            (Path("/project/image.jpg"), True),
            (Path("/project/icon.svg"), True),
            (Path("/project/styles.css"), True),
            (Path("/project/theme.scss"), True),
            (Path("/project/main.py"), False),
        ]

        for path, expected_is_resource in resource_cases:
            code_file = CodeFile(
                project_name="test",
                path=path,
                content="",
                language="python",
                size=0,
                last_modified=datetime.now(),
            )
            assert code_file.is_resource == expected_is_resource, f"Failed for {path}"

    def test_documentation_file_detection(self):
        """Test detection of documentation files."""
        doc_cases = [
            (Path("/project/README.md"), True),
            (Path("/project/docs.md"), True),
            (Path("/project/api.rst"), True),
            (Path("/project/guide.adoc"), True),
            (Path("/project/readme.txt"), True),
            (Path("/project/CHANGELOG.md"), True),
            (Path("/project/LICENSE"), True),
            (Path("/project/CONTRIBUTING.md"), True),
            (Path("/project/main.py"), False),
        ]

        for path, expected_is_doc in doc_cases:
            code_file = CodeFile(
                project_name="test",
                path=path,
                content="",
                language="markdown" if path.suffix == ".md" else "text",
                size=0,
                last_modified=datetime.now(),
            )
            assert code_file.is_documentation == expected_is_doc, f"Failed for {path}"

    def test_line_count_calculation(self):
        """Test automatic line count calculation."""
        content = """def function1():
    pass

def function2():
    return 42

class MyClass:
    def method(self):
        pass"""

        code_file = CodeFile(
            project_name="test",
            path=Path("/project/main.py"),
            content=content,
            language="python",
            size=len(content),
            last_modified=datetime.now(),
        )

        assert code_file.line_count == 9

    def test_python_namespace_extraction(self):
        """Test namespace extraction for Python files."""
        # Test with src directory
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/src/mypackage/submodule/main.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
        )
        assert code_file.namespace == "mypackage.submodule"

        # Test with site-packages
        code_file = CodeFile(
            project_name="test",
            path=Path("/usr/lib/python3.9/site-packages/requests/api.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
        )
        assert code_file.namespace == "requests"

    def test_java_namespace_extraction(self):
        """Test namespace extraction for Java files."""
        content = """package com.example.app;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}"""

        code_file = CodeFile(
            project_name="test",
            path=Path("/project/src/Main.java"),
            content=content,
            language="java",
            size=len(content),
            last_modified=datetime.now(),
        )

        assert code_file.namespace == "com.example.app"

    def test_csharp_namespace_extraction(self):
        """Test namespace extraction for C# files."""
        content = """using System;

namespace MyApp.Services
{
    public class UserService
    {
        public void DoSomething() { }
    }
}"""

        code_file = CodeFile(
            project_name="test",
            path=Path("/project/UserService.cs"),
            content=content,
            language="csharp",
            size=len(content),
            last_modified=datetime.now(),
        )

        assert code_file.namespace == "MyApp.Services"

    def test_typescript_namespace_extraction(self):
        """Test namespace extraction for TypeScript files."""
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/src/components/ui/Button.tsx"),
            content="",
            language="typescript",
            size=0,
            last_modified=datetime.now(),
        )

        assert code_file.namespace == "components/ui"

    def test_file_category_property(self):
        """Test the file_category property."""
        base_args = {
            "project_name": "test",
            "path": Path("/project/file.py"),
            "content": "",
            "language": "python",
            "size": 0,
            "last_modified": datetime.now(),
        }

        # Test file
        code_file = CodeFile(**{**base_args, "is_test": True})
        assert code_file.file_category == "test"

        # Config file
        code_file = CodeFile(**{**base_args, "is_config": True})
        assert code_file.file_category == "config"

        # Resource file
        code_file = CodeFile(**{**base_args, "is_resource": True})
        assert code_file.file_category == "resource"

        # Documentation file
        code_file = CodeFile(**{**base_args, "is_documentation": True})
        assert code_file.file_category == "documentation"

        # Source file (default)
        code_file = CodeFile(**base_args)
        assert code_file.file_category == "source"

    def test_file_hash_property(self):
        """Test the file_hash property."""
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/main.py"),
            content="def main():\n    pass",
            language="python",
            size=22,
            last_modified=datetime.now(),
        )

        # Hash should be consistent
        hash1 = code_file.file_hash
        hash2 = code_file.file_hash
        assert hash1 == hash2

        # Hash should be a valid hex string
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_explicit_type_flags_override_detection(self):
        """Test that explicit type flags override auto-detection."""
        # Even though filename suggests config, explicit flag says it's a test
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/config.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
            is_test=True,
        )

        assert code_file.is_test is True
        assert code_file.is_config is False  # Auto-detection was skipped
        assert code_file.file_category == "test"

    def test_optional_metadata_fields(self):
        """Test optional metadata fields."""
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/main.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
            package_path="/project",
            module_imports=["os", "sys", "datetime"],
            exported_symbols=["main", "helper_function", "MyClass"],
            dependencies=["requests", "numpy", "pandas"],
            complexity_score=15,
        )

        assert code_file.package_path == "/project"
        assert code_file.module_imports == ["os", "sys", "datetime"]
        assert code_file.exported_symbols == ["main", "helper_function", "MyClass"]
        assert code_file.dependencies == ["requests", "numpy", "pandas"]
        assert code_file.complexity_score == 15

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty content
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/empty.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
        )
        assert code_file.line_count == 1  # Empty string has 1 "line"

        # Single line content
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/oneline.py"),
            content="print('hello')",
            language="python",
            size=14,
            last_modified=datetime.now(),
        )
        assert code_file.line_count == 1

        # Path without src directory for namespace extraction
        code_file = CodeFile(
            project_name="test",
            path=Path("/project/main.py"),
            content="",
            language="python",
            size=0,
            last_modified=datetime.now(),
        )
        assert code_file.namespace is None
