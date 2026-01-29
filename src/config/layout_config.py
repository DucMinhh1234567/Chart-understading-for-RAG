"""
Layout configuration for chart region detection.

Centralizes all magic numbers used in chart detection and OCR.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChartLayoutConfig:
    """
    Configuration for chart layout detection.
    
    All values are ratios (0.0-1.0) relative to image dimensions
    unless otherwise noted.
    """
    
    # ==================== Title Region ====================
    # Title is typically at the top of the image
    TITLE_REGION_MAX_Y: float = 0.3  # Top 30%
    
    # ==================== Y-axis Label Region ====================
    # Y-axis label is typically on the left side, rotated 90 degrees
    YLABEL_REGION_MAX_X: float = 0.3  # Left 30%
    YLABEL_REGION_MIN_Y: float = 0.15  # Not too close to top
    YLABEL_REGION_MAX_Y: float = 0.85  # Not too close to bottom
    YLABEL_CENTER_Y: float = 0.45  # Ideal center position
    YLABEL_NOT_TITLE_MIN_Y: float = 0.25  # Must be below title area
    YLABEL_NOT_CATEGORY_MAX_Y: float = 0.75  # Must be above category area
    
    # ==================== X-axis Label Region ====================
    # X-axis label is typically at the bottom center
    XLABEL_REGION_MIN_Y: float = 0.85  # Bottom 15%
    XLABEL_REGION_MIN_X: float = 0.3  # Not too far left
    
    # ==================== Category Region ====================
    # Categories are below the bars, above x-axis label
    CATEGORY_REGION_MIN_Y: float = 0.6  # Lower 40%
    
    # ==================== Y-axis Ticks Region ====================
    # Y-axis tick labels are on the left
    YTICK_REGION_MAX_X: float = 0.2  # Left 20%
    
    # ==================== Left Region for YLabel Extraction ====================
    # Region to crop for dedicated ylabel OCR
    LEFT_REGION_MAX_X: float = 0.15  # Left 15%
    LEFT_REGION_MIN_Y: float = 0.2  # Skip top 20%
    LEFT_REGION_MAX_Y: float = 0.8  # Skip bottom 20%
    
    # ==================== Spacing and Distance Thresholds ====================
    # Absolute pixel values
    MIN_SPACING_THRESHOLD: int = 40  # Minimum spacing for category matching
    SPACING_FACTOR: float = 2.5  # Dynamic spacing factor
    LINE_GROUPING_TOLERANCE: int = 20  # Pixels for grouping text on same line
    
    # ==================== Value Label Matching ====================
    # Thresholds for matching value labels to bars
    VALUE_X_THRESHOLD: int = 50  # Max horizontal distance
    VALUE_Y_THRESHOLD: int = 80  # Max vertical distance
    VALUE_TOTAL_DIST_THRESHOLD: int = 60  # Max weighted total distance
    VALUE_Y_WEIGHT: float = 0.5  # Weight for y-distance in total
    
    # ==================== Vertical Text Detection ====================
    VERTICAL_ASPECT_RATIO: float = 1.5  # height/width > 1.5 = vertical
    
    # ==================== OCR Confidence Thresholds ====================
    DEFAULT_CONFIDENCE: float = 0.5  # Default EasyOCR threshold
    LOW_CONFIDENCE: float = 0.2  # Lower threshold for catching more text
    YLABEL_CONFIDENCE: float = 0.3  # Threshold for ylabel extraction
    TESSERACT_CONFIDENCE: int = 50  # Tesseract confidence (0-100)
    
    # ==================== Fallback Values ====================
    FALLBACK_BASELINE_OFFSET: int = 50  # Pixels from bottom for baseline
    FALLBACK_Y_MIN: float = 0.0  # Default y-axis minimum
    FALLBACK_Y_MAX: float = 100.0  # Default y-axis maximum
    
    # ==================== Supported Formats ====================
    SUPPORTED_IMAGE_FORMATS: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    VALID_OCR_METHODS: tuple = ('easyocr', 'tesseract')


# Default configuration instance
DEFAULT_LAYOUT = ChartLayoutConfig()
