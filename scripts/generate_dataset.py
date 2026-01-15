"""
Synthetic bar chart dataset generator for chart-understanding.

Usage (từ thư mục project `bleh/`):
    python -m scripts.generate_dataset

Tạo 500 biểu đồ cột và lưu:
- Ảnh:   data/raw/bar_charts/chart_0001.png, ...
- JSON:  data/annotations/bar_charts.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAR_CHART_DIR = PROJECT_ROOT / "data" / "raw" / "bar_charts"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "bar_charts.json"


TITLES = [
    "Monthly Sales",
    "Student Performance",
    "Product Revenue",
    "Website Traffic",
    "Quarterly Profits",
    "Survey Results",
    "Market Share",
]

XLABELS = [
    "Months",
    "Students",
    "Products",
    "Categories",
    "Regions",
]

YLABELS = [
    "Sales ($)",
    "Scores",
    "Revenue ($)",
    "Visits",
    "Value",
]

COLORMAPS = [
    "tab10",
    "tab20",
    "Set2",
    "Pastel1",
    "Accent",
]


def _make_categories(num_bars: int, xlabel: str) -> List[str]:
    """Tạo danh sách category cho trục X dựa trên nhãn."""
    xlabel_lower = xlabel.lower()

    if "month" in xlabel_lower:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        start = random.randint(0, max(0, len(months) - num_bars))
        return months[start : start + num_bars]

    if "student" in xlabel_lower:
        return [f"Student {chr(ord('A') + i)}" for i in range(num_bars)]

    if "product" in xlabel_lower:
        return [f"Product {i + 1}" for i in range(num_bars)]

    if "region" in xlabel_lower:
        base = ["North", "South", "East", "West", "Central"]
        if num_bars <= len(base):
            return base[:num_bars]

    # Fallback generic categories
    return [f"Cat{i + 1}" for i in range(num_bars)]


def _generate_single_bar_chart(
    index: int,
    total: int,
) -> Dict[str, Any]:
    """
    Sinh một bar chart và trả về metadata.

    Có thể raise exception; caller sẽ catch và xử lý.
    """
    num_bars = random.randint(3, 8)

    title = random.choice(TITLES)
    xlabel = random.choice(XLABELS)
    ylabel = random.choice(YLABELS)
    cmap_name = random.choice(COLORMAPS)

    categories = _make_categories(num_bars, xlabel)
    values = np.random.uniform(low=5.0, high=100.0, size=num_bars).round(2).tolist()

    show_grid = bool(random.getrandbits(1))
    show_value_labels = bool(random.getrandbits(1))

    # Tên file: chart_0001.png, ...
    filename = f"chart_{index:04d}.png"
    rel_image_path = Path("data") / "raw" / "bar_charts" / filename
    abs_image_path = BAR_CHART_DIR / filename

    # Vẽ chart
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    x = np.arange(num_bars)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.2, 0.8, num_bars))

    bars = ax.bar(x, values, color=colors)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")

    if show_grid:
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    if show_value_labels:
        for rect, val in zip(bars, values):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    abs_image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(abs_image_path)
    plt.close(fig)

    metadata: Dict[str, Any] = {
        "image": str(rel_image_path.as_posix()),
        "metadata": {
            "type": "bar_chart",
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "categories": categories,
            "values": values,
            "options": {
                "grid": show_grid,
                "value_labels": show_value_labels,
                "colormap": cmap_name,
            },
        },
    }

    # Progress indicator cho từng chart
    print(f"[{index}/{total}] Generated {rel_image_path}", flush=True)

    return metadata


def generate_bar_chart_dataset(num_charts: int = 500) -> None:
    """Generate toàn bộ dataset bar chart synthetic."""
    BAR_CHART_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict[str, Any]] = []
    errors: List[str] = []

    print(f"Generating {num_charts} bar charts into {BAR_CHART_DIR} ...")

    for i in range(1, num_charts + 1):
        try:
            meta = _generate_single_bar_chart(i, num_charts)
            annotations.append(meta)
        except Exception as exc:  # noqa: BLE001
            msg = f"Error generating chart {i}: {exc}"
            errors.append(msg)
            print(msg, file=sys.stderr, flush=True)

    # Ghi annotations JSON
    try:
        with ANNOTATION_FILE.open("w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        print(f"\nSaved annotations to {ANNOTATION_FILE}")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write annotation file: {exc}", file=sys.stderr)

    if errors:
        print(f"\nCompleted with {len(errors)} error(s). See stderr for details.")
    else:
        print("\nCompleted without errors.")


def main(argv: List[str] | None = None) -> int:
    """Entry point cho CLI đơn giản."""
    _ = argv  # hiện tại chưa parse args, để mở rộng sau
    generate_bar_chart_dataset(num_charts=500)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

