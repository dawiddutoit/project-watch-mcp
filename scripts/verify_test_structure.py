#!/usr/bin/env python3
"""Verify that test structure mirrors source structure."""

import sys
from pathlib import Path


def get_source_structure(src_dir):
    """Get the structure of source modules."""
    modules = {}
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            relative = py_file.relative_to(src_dir)
            # Skip __init__.py files
            if relative.name != "__init__.py":
                modules[str(relative)] = py_file
    return modules


def get_test_structure(test_dir):
    """Get the structure of test modules."""
    tests = {}
    for py_file in test_dir.rglob("test_*.py"):
        if "__pycache__" not in str(py_file):
            relative = py_file.relative_to(test_dir)
            # Extract the module name from test file name
            module_path = str(relative).replace("test_", "", 1)
            tests[module_path] = py_file
    return tests


def verify_structure():
    """Verify test structure mirrors source structure."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "project_watch_mcp"
    test_dir = project_root / "tests" / "unit"
    
    # Known exceptions - modules that don't need tests
    SKIP_TEST_FOR = [
        "complexity_analysis/languages/kotlin_analyzer_old.py",  # Deprecated
        "vector_search/neo4j_native_vectors_old.py",  # Deprecated
        "complexity_analysis/models.py",  # Data models only
        "language_detection/models.py",  # Data models only
    ]
    
    # Known special test files that don't map 1:1 to source
    SPECIAL_TEST_FILES = [
        "cli_initialize.py",  # Tests cli.py initialization
        "cli_monitoring.py",  # Tests cli.py monitoring
        "mcp_server.py",  # Tests server.py MCP functionality
        "lucene_escaping.py",  # Tests neo4j_rag.py Lucene escaping
        "return_type_validation.py",  # General validation tests
    ]
    
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return False
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return False
    
    source_modules = get_source_structure(src_dir)
    test_modules = get_test_structure(test_dir)
    
    # Check for source modules without tests
    missing_tests = []
    for module_path in source_modules:
        # Skip known exceptions
        if module_path in SKIP_TEST_FOR:
            continue
            
        # Special cases where test names don't exactly match
        test_variations = [
            module_path,  # Exact match
            module_path.replace("__main__.py", "main_module.py"),  # __main__ special case
        ]
        
        # Check if any variation exists
        has_test = any(var in test_modules for var in test_variations)
        
        # Also check for comprehensive/extended test files
        base_name = Path(module_path).stem
        parent_dir = Path(module_path).parent
        comprehensive_tests = [
            str(parent_dir / f"{base_name}_comprehensive.py"),
            str(parent_dir / f"{base_name}_extended.py"),
        ]
        has_comprehensive = any(comp in test_modules for comp in comprehensive_tests)
        
        if not has_test and not has_comprehensive:
            missing_tests.append(module_path)
    
    # Check for misplaced tests (tests in root that should be in subdirs)
    misplaced_tests = []
    for test_path, test_file in test_modules.items():
        # Get the directory of the test
        test_dir_path = Path(test_path).parent
        
        # If test is for a module in a subdirectory but test is in root
        if str(test_dir_path) == "." and "/" in test_path:
            # Check if this maps to a source file in root
            potential_source = test_path.replace("_comprehensive.py", ".py").replace("_extended.py", ".py")
            if potential_source not in source_modules:
                # Check for special cases
                if not any(src.endswith(Path(potential_source).name) for src in source_modules):
                    misplaced_tests.append(test_path)
    
    # Report results
    print("=" * 60)
    print("Test Structure Verification Report")
    print("=" * 60)
    
    print(f"\nSource modules found: {len(source_modules)}")
    print(f"Test modules found: {len(test_modules)}")
    
    if missing_tests:
        print(f"\n⚠️  Source modules without tests ({len(missing_tests)}):")
        for module in sorted(missing_tests):
            print(f"  - {module}")
    else:
        print("\n✅ All source modules have corresponding tests!")
    
    if misplaced_tests:
        print(f"\n⚠️  Potentially misplaced test files ({len(misplaced_tests)}):")
        for test in sorted(misplaced_tests):
            print(f"  - {test}")
    else:
        print("\n✅ All test files appear to be correctly placed!")
    
    # List test files in root directory (should only be for root source files)
    root_tests = [t for t in test_modules if "/" not in t]
    if root_tests:
        print(f"\nTest files in root directory ({len(root_tests)}):")
        for test in sorted(root_tests):
            # Check if corresponding source exists
            source_name = test.replace("test_", "", 1)
            source_name = source_name.replace("main_module.py", "__main__.py")
            if source_name in source_modules:
                print(f"  ✅ {test} -> {source_name}")
            else:
                # Check for comprehensive/extended tests
                base = test.replace("_comprehensive.py", ".py").replace("_extended.py", ".py")
                base = base.replace("test_", "", 1)
                base = base.replace("main_module.py", "__main__.py")
                if base in source_modules:
                    print(f"  ✅ {test} -> {base} (extended)")
                elif test in SPECIAL_TEST_FILES:
                    print(f"  ℹ️  {test} -> Special test file")
                else:
                    print(f"  ⚠️  {test} -> NO MATCHING SOURCE")
    
    print("\n" + "=" * 60)
    
    return len(missing_tests) == 0 and len(misplaced_tests) == 0


if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)