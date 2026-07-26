"""Focused multi-objective sequential ranking track.

This package is intentionally independent from the optional control-plane demo.  It owns the
single resume-facing path: KuaiRand -> temporal sequences -> model comparison -> export.
"""

from streamrank.focused.runner import run_experiment

__all__ = ["run_experiment"]
