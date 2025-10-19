"""Tests for package initialization."""
import sys
from io import StringIO

import eclipsera


def test_version():
    """Test version is accessible."""
    assert hasattr(eclipsera, "__version__")
    assert isinstance(eclipsera.__version__, str)


def test_show_versions(monkeypatch):
    """Test show_versions function."""
    # Capture stdout
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    
    eclipsera.show_versions()
    
    output = captured_output.getvalue()
    
    # Check key components are in output
    assert "Eclipsera version" in output
    assert "Python version" in output
    assert "numpy" in output
    assert "scipy" in output


def test_imports():
    """Test that key modules are importable."""
    from eclipsera import core, ml, preprocessing, model_selection, pipeline
    
    assert core is not None
    assert ml is not None
    assert preprocessing is not None
    assert model_selection is not None
    assert pipeline is not None


def test_module_attributes():
    """Test module-level attributes."""
    assert hasattr(eclipsera, "__author__")
    assert hasattr(eclipsera, "__email__")
    assert hasattr(eclipsera, "__license__")
    assert eclipsera.__license__ == "MIT"
