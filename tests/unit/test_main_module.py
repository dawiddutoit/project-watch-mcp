"""Test the __main__ module."""

from unittest.mock import patch


def test_main_module():
    """Test the __main__ module execution."""
    with patch("project_watch_mcp.__main__.cli") as mock_cli:
        # Import should trigger the if __name__ == "__main__" block

        # Since we're importing, __name__ won't be __main__, so we need to call directly
        from project_watch_mcp.__main__ import cli

        cli()
        mock_cli.assert_called_once()
