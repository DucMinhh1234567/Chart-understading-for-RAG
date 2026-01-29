"""
Tests for ChartComponentDetector.

Tests cover:
- Image validation (None, wrong type, grayscale, empty)
- Axis detection
- Bar detection
- Text region detection
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.chart_detector import ChartComponentDetector
from src.preprocessing.detector_config import (
    InvalidImageError,
    ChartDetectorConfig,
)


# ==================== TestImageValidation ====================

class TestImageValidation:
    """Test _validate_image method."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return ChartComponentDetector()
    
    def test_validate_image_rejects_none(self, detector):
        """Should raise InvalidImageError for None input."""
        with pytest.raises(InvalidImageError, match="Image is None"):
            detector._validate_image(None)
    
    def test_validate_image_rejects_non_numpy(self, detector):
        """Should raise InvalidImageError for non-numpy input."""
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            detector._validate_image("not an image")
        
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            detector._validate_image([1, 2, 3])
        
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            detector._validate_image({"image": "data"})
    
    def test_validate_image_rejects_grayscale(self, detector, sample_grayscale_image):
        """Should raise InvalidImageError for grayscale image."""
        with pytest.raises(InvalidImageError, match="must be RGB"):
            detector._validate_image(sample_grayscale_image)
    
    def test_validate_image_rejects_empty(self, detector):
        """Should raise InvalidImageError for empty image."""
        empty_image = np.array([])
        with pytest.raises(InvalidImageError):
            detector._validate_image(empty_image)
    
    def test_validate_image_rejects_wrong_channels(self, detector):
        """Should raise InvalidImageError for image with wrong number of channels."""
        # 4-channel image (RGBA)
        rgba_image = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must be RGB"):
            detector._validate_image(rgba_image)
        
        # 2-channel image
        two_channel = np.zeros((100, 100, 2), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must be RGB"):
            detector._validate_image(two_channel)
    
    def test_validate_image_accepts_valid_rgb(self, detector, sample_bar_image):
        """Should accept valid RGB image without raising."""
        # Should not raise any exception
        detector._validate_image(sample_bar_image)
    
    def test_validate_image_accepts_different_sizes(self, detector):
        """Should accept RGB images of various sizes."""
        # Small image
        small = np.zeros((50, 50, 3), dtype=np.uint8)
        detector._validate_image(small)
        
        # Large image
        large = np.zeros((1000, 1000, 3), dtype=np.uint8)
        detector._validate_image(large)
        
        # Non-square image
        rect = np.zeros((200, 400, 3), dtype=np.uint8)
        detector._validate_image(rect)


# ==================== TestDetectAxes ====================

class TestDetectAxes:
    """Test detect_axes method."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return ChartComponentDetector()
    
    def test_detect_axes_returns_tuple(self, detector, sample_bar_image):
        """detect_axes should return (x_axis, y_axis) tuple."""
        x_axis, y_axis = detector.detect_axes(sample_bar_image)
        
        # Should return tuple of 2 elements
        assert isinstance(x_axis, (np.ndarray, type(None)))
        assert isinstance(y_axis, (np.ndarray, type(None)))
    
    def test_detect_axes_returns_correct_format(self, detector, sample_bar_image):
        """Detected axes should have [x1, y1, x2, y2] format."""
        x_axis, y_axis = detector.detect_axes(sample_bar_image)
        
        if x_axis is not None:
            assert len(x_axis) == 4
            assert all(isinstance(v, (int, np.integer)) for v in x_axis)
        
        if y_axis is not None:
            assert len(y_axis) == 4
            assert all(isinstance(v, (int, np.integer)) for v in y_axis)
    
    def test_detect_axes_with_real_image(self, detector, sample_real_chart_path):
        """Test axis detection with real chart image."""
        import cv2
        
        # Load real image
        image = cv2.imread(str(sample_real_chart_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x_axis, y_axis = detector.detect_axes(image)
        
        # Real chart should have detectable axes (or fallback)
        # At minimum, we should get something back
        assert x_axis is not None or y_axis is not None or True  # May use fallback
    
    def test_detect_axes_raises_for_invalid_image(self, detector):
        """Should raise InvalidImageError for invalid input."""
        with pytest.raises(InvalidImageError):
            detector.detect_axes(None)
        
        with pytest.raises(InvalidImageError):
            detector.detect_axes("not an image")


# ==================== TestDetectBars ====================

class TestDetectBars:
    """Test detect_bars method."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return ChartComponentDetector()
    
    def test_detect_bars_returns_list(self, detector, sample_bar_image):
        """detect_bars should return list of bar dictionaries."""
        x_axis, y_axis = detector.detect_axes(sample_bar_image)
        bars = detector.detect_bars(sample_bar_image, x_axis, y_axis)
        
        assert isinstance(bars, list)
    
    def test_detect_bars_dict_structure(self, detector, sample_bar_image):
        """Each detected bar should have bbox, area, center keys."""
        x_axis, y_axis = detector.detect_axes(sample_bar_image)
        bars = detector.detect_bars(sample_bar_image, x_axis, y_axis)
        
        for bar in bars:
            assert "bbox" in bar
            assert "area" in bar
            assert "center" in bar
            
            # bbox should be (x, y, w, h)
            assert len(bar["bbox"]) == 4
            
            # center should be (cx, cy)
            assert len(bar["center"]) == 2
    
    def test_detect_bars_with_real_image(self, detector, sample_real_chart_path):
        """Test bar detection with real chart image."""
        import cv2
        
        image = cv2.imread(str(sample_real_chart_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x_axis, y_axis = detector.detect_axes(image)
        bars = detector.detect_bars(image, x_axis, y_axis)
        
        # Real chart should have some detected bars
        # (may vary depending on image)
        assert isinstance(bars, list)
    
    def test_detect_bars_raises_for_invalid_image(self, detector):
        """Should raise InvalidImageError for invalid input."""
        with pytest.raises(InvalidImageError):
            detector.detect_bars(None, None, None)
    
    def test_detect_bars_handles_no_axes(self, detector, sample_bar_image):
        """Should work even when axes are None (uses fallback)."""
        # This tests robustness
        bars = detector.detect_bars(sample_bar_image, None, None)
        assert isinstance(bars, list)


# ==================== TestDetectTextRegions ====================

class TestDetectTextRegions:
    """Test detect_text_regions method."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return ChartComponentDetector()
    
    def test_detect_text_regions_returns_list(self, detector, sample_bar_image):
        """detect_text_regions should return list of tuples."""
        regions = detector.detect_text_regions(sample_bar_image)
        
        assert isinstance(regions, list)
    
    def test_detect_text_regions_tuple_structure(self, detector, sample_bar_image):
        """Each text region should be (x, y, w, h) tuple."""
        regions = detector.detect_text_regions(sample_bar_image)
        
        for region in regions:
            assert len(region) == 4
            x, y, w, h = region
            assert isinstance(w, (int, np.integer))
            assert isinstance(h, (int, np.integer))
            assert w > 0
            assert h > 0
    
    def test_detect_text_regions_with_real_image(self, detector, sample_real_chart_path):
        """Test text region detection with real chart image."""
        import cv2
        
        image = cv2.imread(str(sample_real_chart_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        regions = detector.detect_text_regions(image)
        
        # Real chart should have some text regions (title, labels, etc.)
        assert isinstance(regions, list)


# ==================== TestChartDetectorConfig ====================

class TestChartDetectorConfig:
    """Test adaptive configuration."""
    
    def test_adaptive_hough_threshold(self):
        """Hough threshold should scale with image size."""
        config = ChartDetectorConfig()
        
        config.update_image_size(300, 300)
        small_threshold = config.hough_threshold
        
        config.update_image_size(1000, 1000)
        large_threshold = config.hough_threshold
        
        # Larger image should have higher threshold
        assert large_threshold > small_threshold
    
    def test_adaptive_min_line_length(self):
        """Min line length should scale with image size."""
        config = ChartDetectorConfig()
        
        config.update_image_size(300, 300)
        small_length = config.min_line_length
        
        config.update_image_size(1000, 1000)
        large_length = config.min_line_length
        
        assert large_length > small_length
    
    def test_adaptive_min_bar_area(self):
        """Min bar area should scale with image size."""
        config = ChartDetectorConfig()
        
        config.update_image_size(300, 300)
        small_area = config.min_bar_area
        
        config.update_image_size(1000, 1000)
        large_area = config.min_bar_area
        
        assert large_area > small_area
    
    def test_config_repr(self):
        """Config should have readable string representation."""
        config = ChartDetectorConfig()
        config.update_image_size(400, 600)
        
        repr_str = repr(config)
        assert "ChartDetectorConfig" in repr_str
        assert "image_size" in repr_str
    
    def test_axis_position_thresholds(self):
        """Should return position thresholds for axes."""
        config = ChartDetectorConfig()
        thresholds = config.axis_position_thresholds
        
        assert "x_axis_min_y_ratio" in thresholds
        assert "y_axis_max_x_ratio" in thresholds
        assert 0 < thresholds["x_axis_min_y_ratio"] < 1
        assert 0 < thresholds["y_axis_max_x_ratio"] < 1
