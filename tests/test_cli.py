"""
Unit and integration tests for J.A.R.V.I.S. CLI interface.
"""

from click.testing import CliRunner
from cli import cli


def test_cli_help():
    """Verify that CLI help displays cleanly with all registered commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "J.A.R.V.I.S." in result.output
    assert "chat" in result.output
    assert "goal" in result.output
    assert "ats" in result.output
    assert "doc" in result.output
    assert "vision" in result.output
    assert "web" in result.output
    assert "workspace" in result.output


def test_cli_workspace_command():
    """Verify that workspace list command executes cleanly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["workspace"])
    assert result.exit_code == 0
    assert "Workspace Deliverables" in result.output


def test_cli_workspace_clean():
    """Verify that workspace clean command executes cleanly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["workspace", "--clean"])
    assert result.exit_code == 0
    assert "Cleaned" in result.output
