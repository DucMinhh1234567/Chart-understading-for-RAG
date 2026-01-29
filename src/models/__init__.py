"""Models package for chart-understanding.

This package contains ML model wrappers (classifier, text generator, etc.)
and data types for structured data representation.
"""
from .data_types import (
    BoundingBox,
    Bar,
    TextRegion,
    ChartLabels,
    DataPoint,
    BarChartData,
)

__all__ = [
    # Data types
    "BoundingBox",
    "Bar",
    "TextRegion",
    "ChartLabels",
    "DataPoint",
    "BarChartData",
    # Modules
    "classifier",
    "text_generator",
]
