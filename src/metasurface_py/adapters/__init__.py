"""Adapters for external solver data and validation."""

from metasurface_py.adapters.lookup import import_lookup_table, validate_lookup_table
from metasurface_py.adapters.validation import compare_models

__all__ = [
    "compare_models",
    "import_lookup_table",
    "validate_lookup_table",
]
