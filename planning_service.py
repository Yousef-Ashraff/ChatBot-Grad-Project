"""
planning_service.py — thin re-export of the automated planning() function.

planning() is now fully automated (no print/input calls, no thread bridge).
This module exists for import-compatibility with any remaining callers.
"""

from planning import planning  # noqa: F401

__all__ = ["planning"]
