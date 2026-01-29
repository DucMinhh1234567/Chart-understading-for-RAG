"""
Tests for BarChartExtractor.

Tests cover:
- Input validation (file not found, unsupported format, invalid ocr_method)
- _is_number helper method
- _extract_categories with OCR error correction
- extract method structure and output format
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.extraction.bar_extractor import BarChartExtractor
from src.preprocessing.detector_config import InvalidImageError


# ==================== Fixtures ====================

@pytest.fixture
def extractor():
    """Create a BarChartExtractor with mocked OCR to avoid slow initialization."""
    with patch('src.extraction.bar_extractor.OCREngine') as MockOCR:
        mock_ocr = MockOCR.return_value
        mock_ocr.read_text_easyocr.return_value = []
        mock_ocr.read_chart_labels.return_value = {
            'title': 'Test Chart',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'legend': [],
            'values': []
        }
        extractor = BarChartExtractor()
        extractor.ocr = mock_ocr
        yield extractor


@pytest.fixture
def sample_chart_path(tmp_path):
    """Create a temporary chart image for testing."""
    # Create a simple bar chart image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw some bars
    bar_colors = [(66, 133, 244), (52, 168, 83), (251, 188, 5)]
    bar_heights = [150, 200, 120]
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
    
    # Draw axes
    cv2.line(img, (50, baseline_y), (550, baseline_y), (0, 0, 0), 2)
    cv2.line(img, (50, 50), (50, baseline_y), (0, 0, 0), 2)
    
    # Save image
    chart_path = tmp_path / "test_chart.png"
    cv2.imwrite(str(chart_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return chart_path


# ==================== TestIsNumber ====================

class TestIsNumber:
    """Test _is_number helper method."""
    
    @pytest.fixture
    def extractor_for_is_number(self):
        """Create extractor without mocking for _is_number tests."""
        with patch('src.extraction.bar_extractor.OCREngine'):
            return BarChartExtractor()
    
    def test_recognizes_integer(self, extractor_for_is_number):
        """Should recognize integers."""
        assert extractor_for_is_number._is_number("42") is True
        assert extractor_for_is_number._is_number("100") is True
        assert extractor_for_is_number._is_number("0") is True
    
    def test_recognizes_float(self, extractor_for_is_number):
        """Should recognize floats."""
        assert extractor_for_is_number._is_number("3.14") is True
        assert extractor_for_is_number._is_number("0.5") is True
        assert extractor_for_is_number._is_number("100.00") is True
    
    def test_recognizes_negative(self, extractor_for_is_number):
        """Should recognize negative numbers."""
        assert extractor_for_is_number._is_number("-10") is True
        assert extractor_for_is_number._is_number("-3.14") is True
    
    def test_recognizes_percentage(self, extractor_for_is_number):
        """Should recognize percentages."""
        assert extractor_for_is_number._is_number("50%") is True
        assert extractor_for_is_number._is_number("100%") is True
        assert extractor_for_is_number._is_number("3.5%") is True
    
    def test_recognizes_with_comma_separator(self, extractor_for_is_number):
        """Should recognize numbers with comma as thousand separator."""
        assert extractor_for_is_number._is_number("1,000") is True
        assert extractor_for_is_number._is_number("1,000,000") is True
    
    def test_rejects_text(self, extractor_for_is_number):
        """Should reject non-number text."""
        assert extractor_for_is_number._is_number("hello") is False
        assert extractor_for_is_number._is_number("Jan") is False
        assert extractor_for_is_number._is_number("Category") is False
    
    def test_handles_empty_string(self, extractor_for_is_number):
        """Should handle empty string."""
        assert extractor_for_is_number._is_number("") is False
        assert extractor_for_is_number._is_number("   ") is False
    
    def test_handles_mixed_alphanumeric(self, extractor_for_is_number):
        """Should handle text starting with digit."""
        # Starts with digit - considered number-like
        assert extractor_for_is_number._is_number("5th") is True
        assert extractor_for_is_number._is_number("1st") is True


# ==================== TestInputValidation ====================

class TestInputValidation:
    """Test input validation in extract method."""
    
    def test_raises_file_not_found(self, extractor):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extractor.extract("nonexistent_file.png")
    
    def test_raises_for_unsupported_format(self, extractor, tmp_path):
        """Should raise ValueError for unsupported image format."""
        # Create a dummy file with unsupported extension
        unsupported_file = tmp_path / "test.gif"
        unsupported_file.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            extractor.extract(str(unsupported_file))
    
    def test_raises_for_invalid_ocr_method(self, extractor, sample_chart_path):
        """Should raise ValueError for invalid OCR method."""
        with pytest.raises(ValueError, match="Invalid OCR method"):
            extractor.extract(str(sample_chart_path), ocr_method="invalid_ocr")
    
    def test_accepts_png_format(self, extractor, tmp_path):
        """Should accept PNG format."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        png_path = tmp_path / "test.png"
        cv2.imwrite(str(png_path), img)
        
        # Should not raise - just check it doesn't error on format validation
        # (may fail later due to mocked OCR, but format validation should pass)
        try:
            extractor.extract(str(png_path))
        except (ValueError, FileNotFoundError) as e:
            if "Unsupported" in str(e) or "not found" in str(e):
                pytest.fail(f"Format validation failed: {e}")
    
    def test_accepts_jpg_format(self, extractor, tmp_path):
        """Should accept JPG format."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        jpg_path = tmp_path / "test.jpg"
        cv2.imwrite(str(jpg_path), img)
        
        try:
            extractor.extract(str(jpg_path))
        except ValueError as e:
            if "Unsupported" in str(e):
                pytest.fail(f"Format validation failed: {e}")
    
    def test_accepts_jpeg_format(self, extractor, tmp_path):
        """Should accept JPEG format."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        jpeg_path = tmp_path / "test.jpeg"
        cv2.imwrite(str(jpeg_path), img)
        
        try:
            extractor.extract(str(jpeg_path))
        except ValueError as e:
            if "Unsupported" in str(e):
                pytest.fail(f"Format validation failed: {e}")
    
    def test_accepts_bmp_format(self, extractor, tmp_path):
        """Should accept BMP format."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bmp_path = tmp_path / "test.bmp"
        cv2.imwrite(str(bmp_path), img)
        
        try:
            extractor.extract(str(bmp_path))
        except ValueError as e:
            if "Unsupported" in str(e):
                pytest.fail(f"Format validation failed: {e}")


# ==================== TestExtractOutputStructure ====================

class TestExtractOutputStructure:
    """Test that extract returns correct structure."""
    
    def test_returns_dict(self, extractor, sample_chart_path):
        """Extract should return a dictionary."""
        result = extractor.extract(str(sample_chart_path))
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, extractor, sample_chart_path):
        """Result should have all required keys."""
        result = extractor.extract(str(sample_chart_path))
        
        required_keys = ['chart_type', 'title', 'x_axis_label', 'y_axis_label', 'data']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_chart_type_is_bar_chart(self, extractor, sample_chart_path):
        """chart_type should be 'bar_chart'."""
        result = extractor.extract(str(sample_chart_path))
        assert result['chart_type'] == 'bar_chart'
    
    def test_data_is_list(self, extractor, sample_chart_path):
        """data should be a list."""
        result = extractor.extract(str(sample_chart_path))
        assert isinstance(result['data'], list)
    
    def test_data_items_have_category_and_value(self, extractor, sample_chart_path):
        """Each data item should have category and value."""
        result = extractor.extract(str(sample_chart_path))
        
        for item in result['data']:
            assert 'category' in item
            assert 'value' in item


# ==================== TestExtractCategories ====================

class TestExtractCategories:
    """Test _extract_categories method with OCR error correction."""
    
    @pytest.fixture
    def extractor_for_categories(self):
        """Create extractor without full mocking for category tests."""
        with patch('src.extraction.bar_extractor.OCREngine'):
            return BarChartExtractor()
    
    def test_normalizes_month_abbreviations(self, extractor_for_categories):
        """Should normalize month abbreviations."""
        labels = {
            'xlabel': 'Month',
            'values': [
                {'text': 'jan', 'position': (100, 350), 'is_number': False},
                {'text': 'feb', 'position': (200, 350), 'is_number': False},
                {'text': 'mar', 'position': (300, 350), 'is_number': False},
            ]
        }
        bars = [
            {'bbox': (75, 200, 50, 150), 'area': 7500, 'center': (100, 275)},
            {'bbox': (175, 150, 50, 200), 'area': 10000, 'center': (200, 250)},
            {'bbox': (275, 180, 50, 170), 'area': 8500, 'center': (300, 265)},
        ]
        
        categories = extractor_for_categories._extract_categories(labels, bars, 600, 400)
        
        assert 'Jan' in categories
        assert 'Feb' in categories
        assert 'Mar' in categories
    
    def test_corrects_ocr_errors_for_months(self, extractor_for_categories):
        """Should correct common OCR errors for months."""
        labels = {
            'xlabel': 'Month',
            'values': [
                {'text': 'jui', 'position': (100, 350), 'is_number': False},  # Should become Jul
                {'text': 'ju1', 'position': (200, 350), 'is_number': False},  # Should become Jul
            ]
        }
        bars = [
            {'bbox': (75, 200, 50, 150), 'area': 7500, 'center': (100, 275)},
            {'bbox': (175, 150, 50, 200), 'area': 10000, 'center': (200, 250)},
        ]
        
        categories = extractor_for_categories._extract_categories(labels, bars, 600, 400)
        
        assert 'Jul' in categories
    
    def test_generates_fallback_categories(self, extractor_for_categories):
        """Should generate fallback category names when no match found."""
        labels = {
            'xlabel': 'Category',
            'values': []  # No category labels
        }
        bars = [
            {'bbox': (100, 200, 50, 150), 'area': 7500, 'center': (125, 275)},
            {'bbox': (200, 150, 50, 200), 'area': 10000, 'center': (225, 250)},
        ]
        
        categories = extractor_for_categories._extract_categories(labels, bars, 600, 400)
        
        assert len(categories) == 2
        assert 'Category 1' in categories
        assert 'Category 2' in categories
    
    def test_filters_out_numbers(self, extractor_for_categories):
        """Should not include numbers as categories."""
        labels = {
            'xlabel': 'Items',
            'values': [
                {'text': '100', 'position': (100, 350), 'is_number': True},
                {'text': 'Apple', 'position': (200, 350), 'is_number': False},
            ]
        }
        bars = [
            {'bbox': (75, 200, 50, 150), 'area': 7500, 'center': (100, 275)},
            {'bbox': (175, 150, 50, 200), 'area': 10000, 'center': (200, 250)},
        ]
        
        categories = extractor_for_categories._extract_categories(labels, bars, 600, 400)
        
        # First bar should get fallback, second should get 'Apple'
        assert '100' not in categories
        assert 'Apple' in categories


# ==================== TestCalculateBarValues ====================

class TestCalculateBarValues:
    """Test _calculate_bar_values and related methods."""
    
    @pytest.fixture
    def extractor_for_values(self):
        """Create extractor for value calculation tests."""
        with patch('src.extraction.bar_extractor.OCREngine'):
            return BarChartExtractor()
    
    def test_returns_list_of_values(self, extractor_for_values):
        """Should return a list of numeric values."""
        bars = [
            {'bbox': (100, 200, 50, 150), 'area': 7500, 'center': (125, 275)},
            {'bbox': (200, 100, 50, 250), 'area': 12500, 'center': (225, 225)},
        ]
        y_axis = np.array([50, 50, 50, 350])  # x1, y1, x2, y2
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        labels = {'values': []}
        
        values = extractor_for_values._calculate_bar_values(bars, y_axis, image, labels)
        
        assert isinstance(values, list)
        assert len(values) == len(bars)
        assert all(isinstance(v, (int, float)) for v in values)
    
    def test_uses_label_values_when_available(self, extractor_for_values):
        """Should use OCR-detected values when close to bar."""
        bars = [
            {'bbox': (100, 200, 50, 150), 'area': 7500, 'center': (125, 275)},
        ]
        y_axis = np.array([50, 50, 50, 350])
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        labels = {
            'values': [
                {'text': '75', 'position': (125, 190), 'is_number': True}
            ]
        }
        
        values = extractor_for_values._calculate_bar_values(bars, y_axis, image, labels)
        
        assert values[0] == 75.0


# ==================== TestYAxisScaleDetection ====================

class TestYAxisScaleDetection:
    """Test _detect_y_axis_scale method."""
    
    @pytest.fixture
    def extractor_for_scale(self):
        """Create extractor for scale detection tests."""
        with patch('src.extraction.bar_extractor.OCREngine'):
            return BarChartExtractor()
    
    def test_returns_correct_format(self, extractor_for_scale):
        """Should return (y_min, y_max, tick_positions, tick_values)."""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        y_axis = np.array([50, 50, 50, 350])
        labels = {
            'values': [
                {'text': '0', 'position': (30, 350), 'is_number': True},
                {'text': '50', 'position': (30, 200), 'is_number': True},
                {'text': '100', 'position': (30, 50), 'is_number': True},
            ]
        }
        
        result = extractor_for_scale._detect_y_axis_scale(image, y_axis, labels)
        
        assert len(result) == 4
        y_min, y_max, tick_positions, tick_values = result
        assert isinstance(y_min, (int, float))
        assert isinstance(y_max, (int, float))
        assert isinstance(tick_positions, list)
        assert isinstance(tick_values, list)
    
    def test_returns_fallback_when_no_ticks(self, extractor_for_scale):
        """Should return fallback scale when no tick labels found."""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        y_axis = np.array([50, 50, 50, 350])
        labels = {'values': []}
        
        y_min, y_max, tick_positions, tick_values = extractor_for_scale._detect_y_axis_scale(
            image, y_axis, labels
        )
        
        # Fallback should be 0-100
        assert y_min == 0
        assert y_max == 100


# ==================== TestIntegration ====================

class TestIntegration:
    """Integration tests with real chart images."""
    
    def test_extract_with_sample_fixture(self, sample_real_chart_path):
        """Test extraction with real sample chart from fixtures."""
        # This test uses the real OCR engine, so it may be slow
        # Skip if no sample chart available
        if not sample_real_chart_path.exists():
            pytest.skip("Sample chart not found")
        
        # Use real extractor (slow due to EasyOCR initialization)
        extractor = BarChartExtractor()
        result = extractor.extract(str(sample_real_chart_path))
        
        # Basic structure checks
        assert 'chart_type' in result
        assert result['chart_type'] == 'bar_chart'
        assert 'data' in result
        assert isinstance(result['data'], list)
