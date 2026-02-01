"""
Data types for Chart Understanding.

Provides type-safe dataclasses for:
- BoundingBox: Rectangle with computed properties
- Bar: Detected bar with bbox, area, center
- TextRegion: OCR result with text, position, confidence
- ChartLabels: Extracted labels (title, xlabel, ylabel, etc.)
- BarChartData: Final structured output
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any, Dict


@dataclass
class BoundingBox:
    """Rectangle defined by position and dimensions."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center point (cx, cy)."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]) -> "BoundingBox":
        """Create from (x, y, width, height) tuple."""
        return cls(x=t[0], y=t[1], width=t[2], height=t[3])


@dataclass
class Bar:
    """Detected bar in a bar chart."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    center: Tuple[int, int]

    @property
    def x(self) -> int:
        """Left edge x coordinate."""
        return self.bbox[0]

    @property
    def y(self) -> int:
        """Top edge y coordinate."""
        return self.bbox[1]

    @property
    def width(self) -> int:
        """Bar width in pixels."""
        return self.bbox[2]

    @property
    def height(self) -> int:
        """Bar height in pixels."""
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        """Height / width ratio."""
        return self.height / self.width if self.width > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)."""
        return {"bbox": self.bbox, "area": self.area, "center": self.center}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bar":
        """Create from dictionary."""
        return cls(bbox=tuple(d["bbox"]), area=d["area"], center=tuple(d["center"]))


@dataclass
class TextRegion:
    """OCR text detection result."""

    text: str
    position: Tuple[int, int]  # (center_x, center_y)
    confidence: float = 0.0
    bbox: Optional[Any] = None
    is_number: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "position": self.position,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "is_number": self.is_number,
        }


@dataclass
class ChartLabels:
    """Extracted labels from a chart."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    legend: List[str] = field(default_factory=list)
    values: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "legend": self.legend,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChartLabels":
        """Create from dictionary."""
        return cls(
            title=d.get("title"),
            xlabel=d.get("xlabel"),
            ylabel=d.get("ylabel"),
            legend=d.get("legend", []),
            values=d.get("values", []),
        )


@dataclass
class DataPoint:
    """Single data point in a chart."""

    category: str
    value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"category": self.category, "value": self.value}


@dataclass
class BarChartData:
    """Structured output for bar chart extraction."""

    chart_type: str = "bar_chart"
    title: str = "Untitled"
    x_axis_label: str = "X-axis"
    y_axis_label: str = "Y-axis"
    data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (main output format)."""
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "data": self.data,
        }

    def add_data_point(self, category: str, value: float) -> None:
        """Add a data point."""
        self.data.append({"category": category, "value": value})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BarChartData":
        """Create from dictionary."""
        return cls(
            chart_type=d.get("chart_type", "bar_chart"),
            title=d.get("title", "Untitled"),
            x_axis_label=d.get("x_axis_label", "X-axis"),
            y_axis_label=d.get("y_axis_label", "Y-axis"),
            data=d.get("data", []),
        )
