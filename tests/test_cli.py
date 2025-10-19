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


def test_cmd_train(monkeypatch):
    """Test train command."""
    import argparse
    
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    
    args = argparse.Namespace(
        config="config.yaml",
        data="data.csv",
        model="linear"
    )
    
    exit_code = cmd_train(args)
    
    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Training functionality coming soon" in output


def test_cmd_predict(monkeypatch):
    """Test predict command."""
    import argparse
    
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    
    args = argparse.Namespace(
        model="model.pkl",
        data="data.csv",
        output="predictions.csv"
    )
    
    exit_code = cmd_predict(args)
    
    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Prediction functionality coming soon" in output


def test_cmd_evaluate(monkeypatch):
    """Test evaluate command."""
    import argparse
    
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    
    args = argparse.Namespace(
        model="model.pkl",
        data="data.csv"
    )
    
    exit_code = cmd_evaluate(args)
    
    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Evaluation functionality coming soon" in output


def test_main_info_command(monkeypatch):
    """Test main with info command."""
    captured_output = StringIO()
    monkeypatch.setattr(sys, "argv", ["eclipsera", "info"])
    monkeypatch.setattr(sys, "stdout", captured_output)
    
    exit_code = main()
    
    assert exit_code == 0
    output = captured_output.getvalue()
    assert "Eclipsera version" in output
