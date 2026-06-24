"""SIMD-accelerated Rust core for infy, built with pyo3.

This package is optional: every infy module that uses it falls back to a pure
Python implementation when the compiled extension is unavailable.
"""

from .infy_core import *  # noqa: F401,F403
