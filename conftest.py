# conftest.py – project root
# Makes pytest add the project root to sys.path so that
# 'from src.preprocess import ...' works in tests without installing the package.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
