"""Tests for CLI functionality."""

import sys
from io import StringIO

import pytest

from eclipsera.cli.main import cmd_evaluate, cmd_info, cmd_predict, cmd_train, main


def test_main_no_args(monkeypatch):
    """Test main with no arguments shows help."""
    monkeypatch.setattr(sys, "argv", ["eclipsera"])

    exit_code = main()

    assert exit_code == 1


def test_main_version(monkeypatch, capsys):
    """Test version flag."""
    monkeypatch.setattr(sys, "argv", ["eclipsera", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    # Version flag causes SystemExit(0)
    assert exc_info.value.code == 0


def test_cmd_info(monkeypatch):
    """Test info command."""
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)

    exit_code = cmd_info()

    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Eclipsera version" in output


def test_cmd_train_missing_file(monkeypatch):
    """Test train command with missing data file."""
    import argparse
    import os
    import tempfile

    # Create a temp file that doesn't exist
    temp_path = os.path.join(tempfile.gettempdir(), "nonexistent_data.csv")

    args = argparse.Namespace(data=temp_path, target="target", task="classification", output=None)

    exit_code = cmd_train(args)

    # Should fail with non-zero exit code for missing file
    assert exit_code == 1


def test_cmd_predict_missing_model(monkeypatch):
    """Test predict command with missing model file."""
    import argparse
    import os
    import tempfile

    # Create a temp file that doesn't exist
    temp_model = os.path.join(tempfile.gettempdir(), "nonexistent_model.pkl")
    temp_data = os.path.join(tempfile.gettempdir(), "nonexistent_data.csv")

    args = argparse.Namespace(model=temp_model, data=temp_data, output=None)

    exit_code = cmd_predict(args)

    # Should fail with non-zero exit code for missing model
    assert exit_code == 1


def test_cmd_evaluate_missing_files(monkeypatch):
    """Test evaluate command with missing files."""
    import argparse
    import os
    import tempfile

    # Create temp files that don't exist
    temp_model = os.path.join(tempfile.gettempdir(), "nonexistent_model.pkl")
    temp_data = os.path.join(tempfile.gettempdir(), "nonexistent_data.csv")

    args = argparse.Namespace(model=temp_model, data=temp_data, target="target")

    exit_code = cmd_evaluate(args)

    # Should fail with non-zero exit code for missing files
    assert exit_code == 1


def test_main_info_command(monkeypatch):
    """Test main with info command."""
    captured_output = StringIO()
    monkeypatch.setattr(sys, "argv", ["eclipsera", "info"])
    monkeypatch.setattr(sys, "stdout", captured_output)

    exit_code = main()

    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Eclipsera version" in output
