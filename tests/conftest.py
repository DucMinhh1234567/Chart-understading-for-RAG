"""
Pytest fixtures for Chart Understanding tests.

Provides:
- Dummy bar chart images (generated programmatically)
- Real chart image paths
- Mock bar data for validator testing
"""
import pytest
import numpy as np
import cv2
from pathlib import Path


# ==================== PATHS ====================

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CHARTS_DIR = FIXTURES_DIR / "sample_charts"


# ==================== IMAGE FIXTURES ====================

@pytest.fixture
def sample_bar_image():
    """
    Create a dummy bar chart image using numpy/cv2.
    
    Returns:
        RGB numpy array (400, 600, 3) with 4 colored bars
    """
    # Create white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw 4 bars with different heights
    bar_colors = [
        (66, 133, 244),   # Blue
        (52, 168, 83),    # Green
        (251, 188, 5),    # Yellow
        (234, 67, 53),    # Red
    ]
    bar_heights = [150, 200, 120, 180]
    bar_width = 50
    spacing = 100
    start_x = 100
    baseline_y = 350
    
    for i, (color, height) in enumerate(zip(bar_colors, bar_heights)):
        x1 = start_x + i * spacing
        x2 = x1 + bar_width
        y1 = baseline_y - height
        y2 = baseline_y
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # Draw x-axis line
    cv2.line(img, (50, baseline_y), (550, baseline_y), (0, 0, 0), 2)
    
    # Draw y-axis line
    cv2.line(img, (50, 50), (50, baseline_y), (0, 0, 0), 2)
    
    return img


@pytest.fixture
def sample_grayscale_image():
    """Create a grayscale image (invalid for chart detection)."""
    return np.ones((400, 600), dtype=np.uint8) * 128


@pytest.fixture
def sample_empty_image():
    """Create an empty (zero-size) image."""
    return np.array([])


@pytest.fixture
def sample_real_chart_path():
    """
    Path to a real chart image from fixtures.
    
    Returns:
        Path object to sample_chart.png
    """
    chart_path = SAMPLE_CHARTS_DIR / "sample_chart.png"
    if not chart_path.exists():
        pytest.skip(f"Sample chart not found: {chart_path}")
    return chart_path


# ==================== MOCK BAR DATA FIXTURES ====================

@pytest.fixture
def mock_bars():
    """
    4 bars with consistent width (50px), evenly spaced (100px apart).
    
    This represents ideal bar detection output.
    """
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
        {"bbox": (300, 230, 50, 120), "area": 6000, "center": (325, 290)},
        {"bbox": (400, 170, 50, 180), "area": 9000, "center": (425, 260)},
    ]


@pytest.fixture
def mock_bars_with_width_outlier():
    """
    3 bars with width=50 + 1 bar with width=150 (outlier).
    
    The outlier should be filtered by WidthValidator.
    """
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
        {"bbox": (300, 180, 150, 170), "area": 25500, "center": (375, 265)},  # Outlier
        {"bbox": (500, 220, 50, 130), "area": 6500, "center": (525, 285)},
    ]


@pytest.fixture
def mock_bars_with_area_outlier():
    """
    3 bars with normal area (~8000) + 1 tiny bar (area=100).
    
    The tiny bar should be filtered by AreaValidator.
    """
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
        {"bbox": (300, 340, 10, 10), "area": 100, "center": (305, 345)},  # Tiny noise
        {"bbox": (400, 170, 50, 180), "area": 9000, "center": (425, 260)},
    ]


@pytest.fixture
def mock_bars_with_spacing_outlier():
    """
    4 bars where 3 are evenly spaced (100px) but 1 has unusual spacing (300px).
    
    SpacingValidator may filter the outlier if width is also unusual.
    """
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},  # 100px gap
        {"bbox": (300, 180, 50, 170), "area": 8500, "center": (325, 265)},   # 100px gap
        {"bbox": (600, 220, 80, 130), "area": 10400, "center": (640, 285)},  # 315px gap + width outlier
    ]


@pytest.fixture
def mock_bars_with_aspect_ratio_outlier():
    """
    3 normal bars + 1 very flat bar (low aspect ratio).
    
    The flat bar should be filtered by AspectRatioValidator.
    """
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},   # aspect = 3.0
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},  # aspect = 4.0
        {"bbox": (300, 340, 100, 10), "area": 1000, "center": (350, 345)},   # aspect = 0.1 (flat)
        {"bbox": (400, 170, 50, 180), "area": 9000, "center": (425, 260)},   # aspect = 3.6
    ]


@pytest.fixture
def mock_single_bar():
    """Single bar - validators should skip when <2 bars."""
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
    ]


@pytest.fixture
def mock_two_bars():
    """Two bars - SpacingValidator requires >=3 bars."""
    return [
        {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
        {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
    ]


# ==================== EXTRACTION TEST FIXTURES ====================

@pytest.fixture
def temp_image_file(tmp_path, sample_bar_image):
    """
    Save dummy bar image to temp file for extractor tests.
    
    Returns:
        Path object to temporary PNG file
    """
    file_path = tmp_path / "test_chart.png"
    # Convert RGB to BGR for cv2.imwrite
    cv2.imwrite(str(file_path), cv2.cvtColor(sample_bar_image, cv2.COLOR_RGB2BGR))
    return file_path


@pytest.fixture
def temp_invalid_format_file(tmp_path):
    """Create a temp file with unsupported format (.txt)."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is not an image")
    return file_path


@pytest.fixture(scope="module")
def ocr_engine():
    """
    OCREngine instance (module-scoped for performance).
    
    EasyOCR takes time to initialize, so we reuse the same instance.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.extraction.ocr_engine import OCREngine
    return OCREngine(languages=['en'], gpu=False)


@pytest.fixture(scope="module")
def bar_extractor():
    """
    BarChartExtractor instance (module-scoped for performance).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.extraction.bar_extractor import BarChartExtractor
    return BarChartExtractor()
