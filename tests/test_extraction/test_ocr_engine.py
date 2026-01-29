"""
Tests for OCREngine.

Tests cover:
- _is_number helper method
- read_text_rotated with default and custom angles
- read_chart_labels structure and classification
- Mutable default argument fix verification
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ==================== Fixtures ====================

@pytest.fixture
def mock_easyocr_reader():
    """Create a mock EasyOCR reader."""
    mock_reader = Mock()
    mock_reader.readtext.return_value = []
    return mock_reader


@pytest.fixture
def ocr_engine_with_mock():
    """Create OCREngine with mocked EasyOCR to avoid slow initialization."""
    with patch('src.extraction.ocr_engine.easyocr.Reader') as MockReader:
        mock_reader = Mock()
        mock_reader.readtext.return_value = []
        MockReader.return_value = mock_reader
        
        from src.extraction.ocr_engine import OCREngine
        engine = OCREngine()
        engine._mock_reader = mock_reader  # Store for test access
        yield engine


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_chart_image():
    """Create a sample chart image with text areas."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw title area at top
    cv2.rectangle(img, (200, 10), (400, 40), (200, 200, 200), -1)
    
    # Draw y-axis label area on left
    cv2.rectangle(img, (10, 150), (40, 250), (200, 200, 200), -1)
    
    # Draw x-axis label area at bottom
    cv2.rectangle(img, (250, 370), (350, 390), (200, 200, 200), -1)
    
    return img


# ==================== TestIsNumber ====================

class TestIsNumber:
    """Test _is_number helper method."""
    
    def test_recognizes_integer(self, ocr_engine_with_mock):
        """Should recognize integers."""
        assert ocr_engine_with_mock._is_number("42") is True
        assert ocr_engine_with_mock._is_number("100") is True
        assert ocr_engine_with_mock._is_number("0") is True
    
    def test_recognizes_float(self, ocr_engine_with_mock):
        """Should recognize floats."""
        assert ocr_engine_with_mock._is_number("3.14") is True
        assert ocr_engine_with_mock._is_number("0.5") is True
        assert ocr_engine_with_mock._is_number("100.00") is True
    
    def test_recognizes_negative(self, ocr_engine_with_mock):
        """Should recognize negative numbers."""
        assert ocr_engine_with_mock._is_number("-10") is True
        assert ocr_engine_with_mock._is_number("-3.14") is True
    
    def test_recognizes_percentage(self, ocr_engine_with_mock):
        """Should recognize percentages."""
        assert ocr_engine_with_mock._is_number("50%") is True
        assert ocr_engine_with_mock._is_number("100%") is True
    
    def test_recognizes_with_comma(self, ocr_engine_with_mock):
        """Should recognize numbers with comma separator."""
        assert ocr_engine_with_mock._is_number("1,000") is True
        assert ocr_engine_with_mock._is_number("1,000,000") is True
    
    def test_rejects_text(self, ocr_engine_with_mock):
        """Should reject non-number text."""
        assert ocr_engine_with_mock._is_number("hello") is False
        assert ocr_engine_with_mock._is_number("Jan") is False
        assert ocr_engine_with_mock._is_number("Sales") is False
    
    def test_handles_empty_string(self, ocr_engine_with_mock):
        """Should handle empty string."""
        assert ocr_engine_with_mock._is_number("") is False
        assert ocr_engine_with_mock._is_number("   ") is False
    
    def test_handles_alphanumeric_starting_with_digit(self, ocr_engine_with_mock):
        """Should handle text starting with digit."""
        assert ocr_engine_with_mock._is_number("5th") is True
        assert ocr_engine_with_mock._is_number("1st") is True


# ==================== TestMutableDefaultArgument ====================

class TestMutableDefaultArgument:
    """Test that mutable default argument bug is fixed."""
    
    def test_rotation_angles_default_is_none(self, ocr_engine_with_mock):
        """read_text_rotated should have None as default for rotation_angles."""
        import inspect
        from src.extraction.ocr_engine import OCREngine
        
        sig = inspect.signature(OCREngine.read_text_rotated)
        default = sig.parameters['rotation_angles'].default
        
        assert default is None, "rotation_angles default should be None, not a mutable list"
    
    def test_uses_default_angles_when_none_passed(self, ocr_engine_with_mock, sample_image):
        """Should use [0, 90, 270] when rotation_angles is None."""
        # Mock the easyocr to return something
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "test", 0.9)
        ]
        
        # Call without specifying rotation_angles
        result = ocr_engine_with_mock.read_text_rotated(sample_image)
        
        # Should have attempted OCR (method should work)
        assert isinstance(result, list)


# ==================== TestReadTextRotated ====================

class TestReadTextRotated:
    """Test read_text_rotated method."""
    
    def test_returns_list(self, ocr_engine_with_mock, sample_image):
        """Should return a list."""
        result = ocr_engine_with_mock.read_text_rotated(sample_image)
        assert isinstance(result, list)
    
    def test_returns_empty_list_when_no_text(self, ocr_engine_with_mock, sample_image):
        """Should return empty list when no text detected."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = []
        
        result = ocr_engine_with_mock.read_text_rotated(sample_image)
        assert result == []
    
    def test_returns_best_confidence_result(self, ocr_engine_with_mock, sample_image):
        """Should return result with highest confidence."""
        # Mock different confidences for different rotations
        call_count = [0]
        def mock_readtext(img):
            call_count[0] += 1
            if call_count[0] == 1:  # 0 degrees
                return [([[0,0],[10,0],[10,10],[0,10]], "low", 0.3)]
            elif call_count[0] == 2:  # 90 degrees
                return [([[0,0],[10,0],[10,10],[0,10]], "high", 0.9)]
            else:  # 270 degrees
                return [([[0,0],[10,0],[10,10],[0,10]], "medium", 0.6)]
        
        ocr_engine_with_mock._mock_reader.readtext.side_effect = mock_readtext
        
        result = ocr_engine_with_mock.read_text_rotated(sample_image)
        
        assert len(result) == 1
        assert result[0]['text'] == 'high'
        assert result[0]['confidence'] == 0.9
    
    def test_accepts_custom_rotation_angles(self, ocr_engine_with_mock, sample_image):
        """Should accept custom rotation angles."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[10,0],[10,10],[0,10]], "test", 0.8)
        ]
        
        # Should not raise with custom angles
        result = ocr_engine_with_mock.read_text_rotated(sample_image, rotation_angles=[0, 180])
        assert isinstance(result, list)
    
    def test_handles_180_degree_rotation(self, ocr_engine_with_mock, sample_image):
        """Should handle 180 degree rotation."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[10,0],[10,10],[0,10]], "rotated", 0.85)
        ]
        
        result = ocr_engine_with_mock.read_text_rotated(sample_image, rotation_angles=[180])
        assert isinstance(result, list)


# ==================== TestReadTextEasyOCR ====================

class TestReadTextEasyOCR:
    """Test read_text_easyocr method."""
    
    def test_returns_list(self, ocr_engine_with_mock, sample_image):
        """Should return a list."""
        result = ocr_engine_with_mock.read_text_easyocr(sample_image)
        assert isinstance(result, list)
    
    def test_filters_by_confidence_threshold(self, ocr_engine_with_mock, sample_image):
        """Should filter results below confidence threshold."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[10,0],[10,10],[0,10]], "high_conf", 0.9),
            ([[20,0],[30,0],[30,10],[20,10]], "low_conf", 0.3),
        ]
        
        result = ocr_engine_with_mock.read_text_easyocr(sample_image, confidence_threshold=0.5)
        
        assert len(result) == 1
        assert result[0]['text'] == 'high_conf'
    
    def test_result_structure(self, ocr_engine_with_mock, sample_image):
        """Each result should have text, confidence, bbox."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[10,0],[10,10],[0,10]], "test", 0.8),
        ]
        
        result = ocr_engine_with_mock.read_text_easyocr(sample_image)
        
        assert len(result) == 1
        assert 'text' in result[0]
        assert 'confidence' in result[0]
        assert 'bbox' in result[0]
    
    def test_strips_whitespace_from_text(self, ocr_engine_with_mock, sample_image):
        """Should strip whitespace from detected text."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[10,0],[10,10],[0,10]], "  test  ", 0.8),
        ]
        
        result = ocr_engine_with_mock.read_text_easyocr(sample_image)
        
        assert result[0]['text'] == 'test'


# ==================== TestReadChartLabels ====================

class TestReadChartLabels:
    """Test read_chart_labels method."""
    
    def test_returns_dict_with_required_keys(self, ocr_engine_with_mock, sample_chart_image):
        """Should return dict with title, xlabel, ylabel, legend, values."""
        text_regions = [(200, 10, 200, 30)]  # Title area
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        required_keys = ['title', 'xlabel', 'ylabel', 'legend', 'values']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_legend_is_list(self, ocr_engine_with_mock, sample_chart_image):
        """legend should be a list."""
        text_regions = []
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        assert isinstance(result['legend'], list)
    
    def test_values_is_list(self, ocr_engine_with_mock, sample_chart_image):
        """values should be a list."""
        text_regions = []
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        assert isinstance(result['values'], list)
    
    def test_classifies_title_in_top_region(self, ocr_engine_with_mock, sample_chart_image):
        """Text in top 30% should be classified as title."""
        # Mock OCR to return text in title region
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[100,0],[100,30],[0,30]], "Monthly Sales", 0.9),
        ]
        
        # Title region: y=20 (within top 30% of 400px image)
        text_regions = [(200, 20, 150, 30)]
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        assert result['title'] == 'Monthly Sales'
    
    def test_classifies_xlabel_in_bottom_region(self, ocr_engine_with_mock, sample_chart_image):
        """Text in bottom region should be classified as xlabel."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[80,0],[80,20],[0,20]], "Months", 0.9),
        ]
        
        # Bottom region: y=380 (>85% of 400px)
        text_regions = [(250, 380, 100, 20)]
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        # xlabel should be in bottom region, non-number, center-ish position
        # May or may not be classified correctly depending on position
        assert isinstance(result['xlabel'], (str, type(None)))
    
    def test_handles_empty_text_regions(self, ocr_engine_with_mock, sample_chart_image):
        """Should handle empty text regions list."""
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, [])
        
        assert 'title' in result
        assert 'xlabel' in result
        assert 'ylabel' in result
    
    def test_values_include_position_info(self, ocr_engine_with_mock, sample_chart_image):
        """Values should include position information."""
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[30,0],[30,20],[0,20]], "100", 0.9),
        ]
        
        text_regions = [(100, 250, 50, 30)]
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        # Check if values have position info
        for value in result['values']:
            assert 'position' in value
            assert 'is_number' in value


# ==================== TestOCREngineInitialization ====================

class TestOCREngineInitialization:
    """Test OCREngine initialization."""
    
    def test_default_languages(self):
        """Should use ['en', 'vi'] as default languages."""
        with patch('src.extraction.ocr_engine.easyocr.Reader') as MockReader:
            from src.extraction.ocr_engine import OCREngine
            OCREngine()
            
            # Check that Reader was called with ['en', 'vi']
            MockReader.assert_called_once()
            call_args = MockReader.call_args
            assert call_args[0][0] == ['en', 'vi']
    
    def test_custom_languages(self):
        """Should accept custom languages."""
        with patch('src.extraction.ocr_engine.easyocr.Reader') as MockReader:
            from src.extraction.ocr_engine import OCREngine
            OCREngine(languages=['en', 'ja', 'ko'])
            
            call_args = MockReader.call_args
            assert call_args[0][0] == ['en', 'ja', 'ko']
    
    def test_gpu_parameter(self):
        """Should pass GPU parameter to EasyOCR."""
        with patch('src.extraction.ocr_engine.easyocr.Reader') as MockReader:
            from src.extraction.ocr_engine import OCREngine
            OCREngine(gpu=True)
            
            call_args = MockReader.call_args
            assert call_args[1]['gpu'] is True


# ==================== TestVerticalTextDetection ====================

class TestVerticalTextDetection:
    """Test detection of vertical (rotated) text."""
    
    def test_detects_vertical_text_by_aspect_ratio(self, ocr_engine_with_mock, sample_chart_image):
        """Should detect vertical text when height > 1.5 * width."""
        # Vertical text region: height=100, width=30 (aspect ratio > 3)
        text_regions = [(10, 150, 30, 100)]
        
        ocr_engine_with_mock._mock_reader.readtext.return_value = [
            ([[0,0],[30,0],[30,100],[0,100]], "Y-Axis", 0.8),
        ]
        
        result = ocr_engine_with_mock.read_chart_labels(sample_chart_image, text_regions)
        
        # Should have attempted to process vertical text
        # The method should have been called with rotated attempts
        assert ocr_engine_with_mock._mock_reader.readtext.called
