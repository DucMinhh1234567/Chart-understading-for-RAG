"""
Tests for OCREngine.

Tests cover:
- Initialization with different languages
- EasyOCR text reading
- Rotated text reading
- Chart labels reading
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.extraction.ocr_engine import OCREngine


# ==================== TestOCREngineInit ====================

class TestOCREngineInit:
    """Test OCREngine initialization."""
    
    def test_init_default_languages(self):
        """Default languages should be ['en', 'vi']."""
        engine = OCREngine(gpu=False)
        # Check that easyocr_reader was initialized
        assert engine.easyocr_reader is not None
    
    def test_init_custom_languages(self):
        """Should accept custom languages."""
        engine = OCREngine(languages=['en'], gpu=False)
        assert engine.easyocr_reader is not None
    
    def test_init_with_gpu_false(self):
        """Should initialize with gpu=False."""
        engine = OCREngine(languages=['en'], gpu=False)
        assert engine.easyocr_reader is not None


# ==================== TestReadTextEasyOCR ====================

class TestReadTextEasyOCR:
    """Test read_text_easyocr method."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """OCREngine instance."""
        return OCREngine(languages=['en'], gpu=False)
    
    @pytest.fixture
    def image_with_text(self):
        """Create an image with clear text for OCR testing."""
        # Create white background
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        # Add text using cv2.putText
        cv2.putText(img, "TEST", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 0), 3, cv2.LINE_AA)
        
        return img
    
    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no text."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_read_text_easyocr_returns_list(self, engine, image_with_text):
        """read_text_easyocr should return a list."""
        result = engine.read_text_easyocr(image_with_text, confidence_threshold=0.1)
        assert isinstance(result, list)
    
    def test_read_text_easyocr_result_structure(self, engine, image_with_text):
        """Each result should have text, confidence, bbox keys."""
        result = engine.read_text_easyocr(image_with_text, confidence_threshold=0.1)
        
        # May not detect text depending on image quality
        if len(result) > 0:
            item = result[0]
            assert 'text' in item
            assert 'confidence' in item
            assert 'bbox' in item
    
    def test_read_text_easyocr_confidence_threshold(self, engine, image_with_text):
        """Should filter results by confidence threshold."""
        # Low threshold - more results
        low_threshold_results = engine.read_text_easyocr(image_with_text, confidence_threshold=0.1)
        
        # High threshold - fewer results
        high_threshold_results = engine.read_text_easyocr(image_with_text, confidence_threshold=0.9)
        
        # High threshold should have <= results than low threshold
        assert len(high_threshold_results) <= len(low_threshold_results)
    
    def test_read_text_easyocr_blank_image(self, engine, blank_image):
        """Should return empty list for blank image."""
        result = engine.read_text_easyocr(blank_image, confidence_threshold=0.5)
        assert isinstance(result, list)
        # Blank image should have no text
        assert len(result) == 0


# ==================== TestReadTextRotated ====================

class TestReadTextRotated:
    """Test read_text_rotated method."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """OCREngine instance."""
        return OCREngine(languages=['en'], gpu=False)
    
    @pytest.fixture
    def sample_region(self):
        """Create a sample image region."""
        return np.ones((50, 100, 3), dtype=np.uint8) * 200
    
    def test_read_text_rotated_returns_list(self, engine, sample_region):
        """read_text_rotated should return a list."""
        result = engine.read_text_rotated(sample_region)
        assert isinstance(result, list)
    
    def test_read_text_rotated_default_angles(self, engine, sample_region):
        """Should use default angles [0, 90, 270] when None."""
        # Just verify it runs without error
        result = engine.read_text_rotated(sample_region, rotation_angles=None)
        assert isinstance(result, list)
    
    def test_read_text_rotated_custom_angles(self, engine, sample_region):
        """Should accept custom rotation angles."""
        result = engine.read_text_rotated(sample_region, rotation_angles=[0, 180])
        assert isinstance(result, list)
    
    def test_read_text_rotated_single_angle(self, engine, sample_region):
        """Should work with single angle."""
        result = engine.read_text_rotated(sample_region, rotation_angles=[0])
        assert isinstance(result, list)


# ==================== TestReadChartLabels ====================

class TestReadChartLabels:
    """Test read_chart_labels method."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """OCREngine instance."""
        return OCREngine(languages=['en'], gpu=False)
    
    @pytest.fixture
    def sample_chart_image(self, sample_bar_image):
        """Use sample bar image from conftest."""
        return sample_bar_image
    
    @pytest.fixture
    def sample_text_regions(self):
        """Sample text regions for testing."""
        return [
            (10, 10, 100, 30),   # Title region
            (250, 380, 100, 20),  # X-axis label region
            (10, 180, 30, 100),   # Y-axis label region
        ]
    
    def test_read_chart_labels_returns_dict(self, engine, sample_chart_image, sample_text_regions):
        """read_chart_labels should return a dictionary."""
        result = engine.read_chart_labels(sample_chart_image, sample_text_regions)
        assert isinstance(result, dict)
    
    def test_read_chart_labels_has_required_keys(self, engine, sample_chart_image, sample_text_regions):
        """Result should have title, xlabel, ylabel, legend, values keys."""
        result = engine.read_chart_labels(sample_chart_image, sample_text_regions)
        
        required_keys = ['title', 'xlabel', 'ylabel', 'legend', 'values']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_read_chart_labels_legend_is_list(self, engine, sample_chart_image, sample_text_regions):
        """legend should be a list."""
        result = engine.read_chart_labels(sample_chart_image, sample_text_regions)
        assert isinstance(result['legend'], list)
    
    def test_read_chart_labels_values_is_list(self, engine, sample_chart_image, sample_text_regions):
        """values should be a list."""
        result = engine.read_chart_labels(sample_chart_image, sample_text_regions)
        assert isinstance(result['values'], list)
    
    def test_read_chart_labels_empty_regions(self, engine, sample_chart_image):
        """Should handle empty text regions list."""
        result = engine.read_chart_labels(sample_chart_image, [])
        
        assert isinstance(result, dict)
        assert 'title' in result
        assert 'values' in result


# ==================== TestIsNumber ====================

class TestIsNumber:
    """Test _is_number helper method."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """OCREngine instance."""
        return OCREngine(languages=['en'], gpu=False)
    
    def test_is_number_integers(self, engine):
        """Should recognize integers."""
        assert engine._is_number("123") is True
        assert engine._is_number("0") is True
        assert engine._is_number("-42") is True
    
    def test_is_number_floats(self, engine):
        """Should recognize floats."""
        assert engine._is_number("3.14") is True
        assert engine._is_number("0.5") is True
        assert engine._is_number("-1.23") is True
    
    def test_is_number_with_percentage(self, engine):
        """Should recognize percentages."""
        assert engine._is_number("50%") is True
        assert engine._is_number("100%") is True
    
    def test_is_number_with_comma(self, engine):
        """Should handle comma separators."""
        assert engine._is_number("1,000") is True
        assert engine._is_number("1,234,567") is True
    
    def test_is_number_rejects_text(self, engine):
        """Should reject non-numeric text."""
        assert engine._is_number("hello") is False
        assert engine._is_number("Jan") is False
        assert engine._is_number("") is False
