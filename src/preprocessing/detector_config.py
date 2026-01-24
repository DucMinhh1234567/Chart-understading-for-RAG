# src/preprocessing/detector_config.py
"""
Configuration system for chart detection with adaptive thresholds.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ==================== CUSTOM EXCEPTIONS ====================

class ChartDetectionError(Exception):
    """Base exception for chart detection failures"""
    pass


class InvalidImageError(ChartDetectionError):
    """Raised when image is invalid (None, wrong shape, etc)"""
    pass


class AxisDetectionError(ChartDetectionError):
    """Raised when axes cannot be detected"""
    pass


class BarDetectionError(ChartDetectionError):
    """Raised when bar detection fails"""
    pass


# ==================== CONFIGURATION CLASS ====================

@dataclass
class ChartDetectorConfig:
    """
    Adaptive configuration for chart detection.
    
    All thresholds are adaptive based on image size unless explicitly overridden.
    This ensures the detector works well across different chart resolutions.
    
    Attributes:
        image_size: Tuple of (height, width) for adaptive thresholds
        chart_type: Type of chart ("vertical_bar", "horizontal_bar", "line")
        
        # Hough Line Transform parameters (for axis detection)
        hough_threshold_override: Override adaptive threshold
        min_line_length_ratio: Minimum line length as ratio of image dimension
        max_line_gap: Maximum gap between line segments
        
        # Bar detection parameters
        min_bar_area_override: Override adaptive minimum bar area
        min_bar_height: Minimum bar height in pixels
        
        # Aspect ratio constraints
        min_aspect_ratio: Minimum height/width ratio for bars
        max_aspect_ratio: Maximum height/width ratio for bars
        
        # Morphological operation parameters
        kernel_sizes: List of kernel sizes for multi-scale detection
        erode_iterations: Number of erosion iterations
        dilate_iterations: Number of dilation iterations
        
        # Bar merging parameters
        merge_max_distance: Maximum distance for merging nearby bars
        overlap_threshold: Overlap ratio for duplicate detection
        
        # Validation parameters
        width_tolerance: Width deviation tolerance (ratio)
        area_tolerance: Area threshold as ratio of median
        spacing_tolerance: Spacing consistency tolerance
        
        # HSV color filtering for bar detection
        hsv_lower_white: Lower HSV threshold for white detection
        hsv_upper_white: Upper HSV threshold for white detection
    """
    
    # Basic properties
    image_size: Optional[Tuple[int, int]] = None
    chart_type: str = "vertical_bar"
    
    # Hough parameters
    hough_threshold_override: Optional[int] = None
    min_line_length_ratio: float = 0.1  # 10% of image dimension
    max_line_gap: int = 10
    
    # Bar detection
    min_bar_area_override: Optional[int] = None
    min_bar_height: int = 5
    
    # Aspect ratio
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 10
    
    # Morphological operations
    kernel_sizes: Tuple[Tuple[int, int], ...] = field(default_factory=lambda: ((3, 3), (5, 5)))
    erode_iterations: int = 1
    dilate_iterations: int = 2
    
    # Merging
    merge_max_distance: int = 30
    overlap_threshold: float = 0.8
    
    # Validation
    width_tolerance_min: float = 0.35  # 35% of median
    width_tolerance_max: float = 2.2   # 220% of median
    area_tolerance: float = 0.12       # 12% of median
    spacing_tolerance: float = 0.4     # 40% variation
    
    # HSV filtering
    hsv_lower_white: Tuple[int, int, int] = (0, 0, 200)
    hsv_upper_white: Tuple[int, int, int] = (180, 30, 255)
    
    # ==================== ADAPTIVE PROPERTIES ====================
    
    @property
    def hough_threshold(self) -> int:
        """
        Adaptive Hough threshold based on image size.
        
        Returns 15% of the minimum image dimension, which scales well:
        - 300x300 image -> threshold = 45
        - 800x600 image -> threshold = 90
        - 1920x1080 image -> threshold = 162
        
        For smaller images, lower threshold catches more lines.
        For larger images, higher threshold filters noise better.
        """
        if self.hough_threshold_override is not None:
            return self.hough_threshold_override
        
        if self.image_size is not None:
            min_dim = min(self.image_size)
            return int(min_dim * 0.15)
        
        # Default fallback
        return 100
    
    @property
    def min_line_length(self) -> int:
        """
        Adaptive minimum line length for Hough transform.
        
        Returns min_line_length_ratio * dimension.
        Default 10% ensures we catch major axes but not noise.
        """
        if self.image_size is not None:
            min_dim = min(self.image_size)
            return int(min_dim * self.min_line_length_ratio)
        
        # Default fallback
        return 100
    
    @property
    def min_bar_area(self) -> int:
        """
        Adaptive minimum bar area based on total image area.
        
        Returns 0.05% of total image area:
        - 800x600 (480K pixels) -> min_area = 240
        - 1920x1080 (2M pixels) -> min_area = 1036
        
        This ensures tiny noise is filtered while detecting small bars.
        """
        if self.min_bar_area_override is not None:
            return self.min_bar_area_override
        
        if self.image_size is not None:
            total_area = self.image_size[0] * self.image_size[1]
            return int(total_area * 0.0005)  # 0.05%
        
        # Default fallback
        return 200
    
    @property
    def axis_position_thresholds(self) -> dict:
        """
        Returns position constraints for axis detection.
        
        X-axis should be in bottom 30% of image.
        Y-axis should be in left 20% of image.
        """
        return {
            'x_axis_min_y_ratio': 0.7,  # Bottom 30%
            'y_axis_max_x_ratio': 0.2,  # Left 20%
        }
    
    @property
    def axis_length_threshold_ratio(self) -> float:
        """
        Minimum axis length as ratio of image dimension.
        
        Axis must span at least 50% of the image to be valid.
        """
        return 0.5
    
    def update_image_size(self, height: int, width: int) -> None:
        """
        Update image size for adaptive thresholds.
        
        Should be called before processing each new image.
        """
        self.image_size = (height, width)
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"ChartDetectorConfig(\n"
            f"  image_size={self.image_size},\n"
            f"  hough_threshold={self.hough_threshold},\n"
            f"  min_line_length={self.min_line_length},\n"
            f"  min_bar_area={self.min_bar_area},\n"
            f"  chart_type='{self.chart_type}'\n"
            f")"
        )
