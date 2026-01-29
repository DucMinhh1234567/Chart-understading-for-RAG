"""
Tests for BarChartExtractor.

Tests cover:
- Input validation (file exists, format, OCR method)
- Output structure (dict keys, data format)
- Integration with real images
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.extraction.bar_extractor import BarChartExtractor
from src.preprocessing.detector_config import InvalidImageError


# ==================== TestExtractValidation ====================

class TestExtractValidation:
    """Test input validation in extract() method."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BarChartExtractor()
    
    def test_extract_raises_file_not_found(self, extractor):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extractor.extract("/nonexistent/path/image.png")
    
    def test_extract_raises_invalid_format_txt(self, extractor, temp_invalid_format_file):
        """Should raise ValueError for .txt file."""
        with pytest.raises(ValueError, match="Unsupported image format"):
            extractor.extract(str(temp_invalid_format_file))
    
    def test_extract_raises_invalid_format_pdf(self, extractor, tmp_path):
        """Should raise ValueError for .pdf file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")  # Minimal PDF header
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            extractor.extract(str(pdf_file))
    
    def test_extract_raises_invalid_ocr_method(self, extractor, temp_image_file):
        """Should raise ValueError for invalid OCR method."""
        with pytest.raises(ValueError, match="Invalid OCR method"):
            extractor.extract(str(temp_image_file), ocr_method='invalid_method')
    
    def test_extract_accepts_valid_ocr_easyocr(self, extractor, temp_image_file):
        """Should accept 'easyocr' as valid OCR method."""
        # Should not raise - just verify it runs
        result = extractor.extract(str(temp_image_file), ocr_method='easyocr')
        assert result is not None
    
    def test_extract_accepts_valid_ocr_tesseract(self, extractor, temp_image_file):
        """Should accept 'tesseract' as valid OCR method."""
        # Should not raise - just verify it runs
        result = extractor.extract(str(temp_image_file), ocr_method='tesseract')
        assert result is not None


# ==================== TestExtractOutput ====================

class TestExtractOutput:
    """Test output structure of extract() method."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BarChartExtractor()
    
    @pytest.fixture
    def extraction_result(self, extractor, temp_image_file):
        """Get extraction result from temp image."""
        return extractor.extract(str(temp_image_file))
    
    def test_extract_returns_dict(self, extraction_result):
        """extract() should return a dictionary."""
        assert isinstance(extraction_result, dict)
    
    def test_extract_has_required_keys(self, extraction_result):
        """Result dict should have all required keys."""
        required_keys = ['chart_type', 'title', 'x_axis_label', 'y_axis_label', 'data']
        for key in required_keys:
            assert key in extraction_result, f"Missing key: {key}"
    
    def test_extract_data_is_list(self, extraction_result):
        """data field should be a list."""
        assert isinstance(extraction_result['data'], list)
    
    def test_extract_chart_type_is_bar_chart(self, extraction_result):
        """chart_type should be 'bar_chart'."""
        assert extraction_result['chart_type'] == 'bar_chart'
    
    def test_extract_title_is_string_or_none(self, extraction_result):
        """title should be a string or None (if no text detected)."""
        # Dummy image may not have detectable text
        assert extraction_result['title'] is None or isinstance(extraction_result['title'], str)
    
    def test_extract_axis_labels_are_strings_or_none(self, extraction_result):
        """x_axis_label and y_axis_label should be strings or None."""
        # Dummy image may not have detectable text
        assert extraction_result['x_axis_label'] is None or isinstance(extraction_result['x_axis_label'], str)
        assert extraction_result['y_axis_label'] is None or isinstance(extraction_result['y_axis_label'], str)


# ==================== TestExtractDataItems ====================

class TestExtractDataItems:
    """Test data items structure."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BarChartExtractor()
    
    def test_extract_data_items_have_category_and_value(self, extractor, temp_image_file):
        """Each data item should have 'category' and 'value' keys."""
        result = extractor.extract(str(temp_image_file))
        
        for i, item in enumerate(result['data']):
            assert 'category' in item, f"Item {i} missing 'category'"
            assert 'value' in item, f"Item {i} missing 'value'"
    
    def test_extract_data_values_are_numeric(self, extractor, temp_image_file):
        """Values should be numeric (int or float)."""
        result = extractor.extract(str(temp_image_file))
        
        for i, item in enumerate(result['data']):
            assert isinstance(item['value'], (int, float)), \
                f"Item {i} value is not numeric: {type(item['value'])}"
    
    def test_extract_data_categories_are_strings(self, extractor, temp_image_file):
        """Categories should be strings."""
        result = extractor.extract(str(temp_image_file))
        
        for i, item in enumerate(result['data']):
            assert isinstance(item['category'], str), \
                f"Item {i} category is not string: {type(item['category'])}"


# ==================== TestExtractWithRealImage ====================

class TestExtractWithRealImage:
    """Test extraction with real chart image."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BarChartExtractor()
    
    def test_extract_with_real_image_returns_data(self, extractor, sample_real_chart_path):
        """Should successfully extract data from real chart image."""
        result = extractor.extract(str(sample_real_chart_path))
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_extract_real_image_has_required_structure(self, extractor, sample_real_chart_path):
        """Real image extraction should have complete structure."""
        result = extractor.extract(str(sample_real_chart_path))
        
        assert 'chart_type' in result
        assert 'title' in result
        assert 'x_axis_label' in result
        assert 'y_axis_label' in result
        assert 'data' in result
    
    def test_extract_real_image_detects_bars(self, extractor, sample_real_chart_path):
        """Should detect at least some bars from real chart."""
        result = extractor.extract(str(sample_real_chart_path))
        
        # Real chart should have some data points
        assert len(result['data']) > 0, "No bars detected from real chart"
    
    def test_extract_real_image_chart_type_correct(self, extractor, sample_real_chart_path):
        """chart_type should be 'bar_chart' for real bar chart image."""
        result = extractor.extract(str(sample_real_chart_path))
        
        assert result['chart_type'] == 'bar_chart'


# ==================== TestHelperMethods ====================

class TestHelperMethods:
    """Test helper methods in BarChartExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BarChartExtractor()
    
    def test_is_number_with_integers(self, extractor):
        """_is_number should recognize integers."""
        assert extractor._is_number("123") is True
        assert extractor._is_number("0") is True
        assert extractor._is_number("-456") is True
    
    def test_is_number_with_floats(self, extractor):
        """_is_number should recognize floats."""
        assert extractor._is_number("123.45") is True
        assert extractor._is_number("0.5") is True
        assert extractor._is_number("-12.34") is True
    
    def test_is_number_with_percentage(self, extractor):
        """_is_number should recognize percentages."""
        assert extractor._is_number("50%") is True
        assert extractor._is_number("100%") is True
    
    def test_is_number_with_comma_separators(self, extractor):
        """_is_number should handle comma separators."""
        assert extractor._is_number("1,000") is True
        assert extractor._is_number("1,234,567") is True
    
    def test_is_number_rejects_text(self, extractor):
        """_is_number should reject pure text."""
        assert extractor._is_number("hello") is False
        assert extractor._is_number("Jan") is False
        assert extractor._is_number("Category A") is False
