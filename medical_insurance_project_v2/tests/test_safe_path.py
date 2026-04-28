"""Tests for the hardened path resolver."""
import pytest
from pathlib import Path
from src.paths import safe_path, DATA_DIR, UPLOAD_DIR

def test_safe_path_ok():
    p = safe_path(UPLOAD_DIR, "myfile.csv")
    assert str(p).startswith(str(UPLOAD_DIR.resolve()))

def test_path_traversal_blocked():
    with pytest.raises(ValueError):
        safe_path(UPLOAD_DIR, "../../etc/passwd")

def test_absolute_path_blocked():
    with pytest.raises(ValueError):
        safe_path(UPLOAD_DIR, "/etc/passwd")
