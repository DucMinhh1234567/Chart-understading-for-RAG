# src/preprocessing/chart_detector.py
"""
Chart component detection with adaptive configuration and robust error handling.

This module provides detection for axes, bars, and text regions in chart images
using computer vision techniques (Hough Transform, connected components, etc.).
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from scipy.spatial import KDTree

from .image_utils import ImagePreprocessor
from .detector_config import (
    ChartDetectorConfig,
    ChartDetectionError,
    InvalidImageError,
    BarDetectionError,
)
from .bar_validators import (
    ValidationPipeline,
    WidthValidator,
    AreaValidator,
    AspectRatioValidator,
    SpacingValidator,
)


logger = logging.getLogger(__name__)


class ChartComponentDetector:
    """
    Detects chart components (axes, bars, text regions) in images.

    Uses adaptive configuration for different image sizes and chart types.
    Includes comprehensive error handling and validation pipeline.
    """

    def __init__(self, preprocessor=None, config: Optional[ChartDetectorConfig] = None):
        """
        Initialize detector with optional configuration.

        Args:
            preprocessor: Image preprocessor instance
            config: ChartDetectorConfig for adaptive thresholds
        """
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.config = config or ChartDetectorConfig()
        self.logger = logging.getLogger(__name__)
        self.validation_pipeline = self._create_validation_pipeline()

        self.logger.info("ChartComponentDetector initialized")

    def _create_validation_pipeline(self) -> ValidationPipeline:
        """
        Create validation pipeline with configured validators.

        Returns:
            ValidationPipeline instance
        """
        validators = [
            WidthValidator(
                min_ratio=self.config.width_tolerance_min,
                max_ratio=self.config.width_tolerance_max,
            ),
            AreaValidator(min_ratio=self.config.area_tolerance, absolute_min=150),
            AspectRatioValidator(
                min_ratio=self.config.min_aspect_ratio,
                max_ratio=self.config.max_aspect_ratio,
            ),
            SpacingValidator(tolerance=self.config.spacing_tolerance),
        ]
        return ValidationPipeline(validators)

    def _validate_image(self, image) -> None:
        """
        Validate that image is suitable for processing.

        Args:
            image: Input image

        Raises:
            InvalidImageError: If image is invalid
        """
        if image is None:
            raise InvalidImageError("Image is None")

        if not isinstance(image, np.ndarray):
            raise InvalidImageError(f"Image must be numpy array, got {type(image)}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise InvalidImageError(
                f"Image must be RGB with shape (H, W, 3), got shape {image.shape}"
            )

        if image.size == 0:
            raise InvalidImageError("Image has zero size")

    def detect_axes(self, image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect X and Y axes using Hough Line Transform with validation.

        Args:
            image: RGB image numpy array

        Returns:
            Tuple of (x_axis, y_axis) where each is [x1, y1, x2, y2] or None

        Raises:
            InvalidImageError: If image is invalid
            ChartDetectionError: If OpenCV operations fail
        """
        # Validate input
        self._validate_image(image)

        # Update config with image size
        h, w = image.shape[:2]
        self.config.update_image_size(h, w)

        self.logger.info(f"Detecting axes for image size: {h}x{w}")
        self.logger.debug(f"Using config: {self.config}")

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Hough Line Transform with adaptive thresholds
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.min_line_length,
                maxLineGap=self.config.max_line_gap,
            )

            if lines is None:
                self.logger.warning("No lines detected by Hough transform")
                return self._fallback_axes(h, w)

            self.logger.debug(f"Detected {len(lines)} lines")

            # Classify lines into horizontal and vertical
            h_lines = []
            v_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 10:  # Horizontal (within 10 degrees)
                    h_lines.append(line[0])
                elif angle > 80:  # Vertical (within 10 degrees)
                    v_lines.append(line[0])

            self.logger.debug(
                f"Classified: {len(h_lines)} horizontal, {len(v_lines)} vertical"
            )

            # Find axes with validation
            x_axis = self._find_bottom_line_validated(h_lines, h, w)
            y_axis = self._find_left_line_validated(v_lines, h, w)

            if x_axis is None:
                self.logger.warning("X-axis not found, using fallback")
            if y_axis is None:
                self.logger.warning("Y-axis not found, using fallback")

            return x_axis, y_axis

        except cv2.error as e:
            self.logger.error(f"OpenCV error in axis detection: {e}")
            raise ChartDetectionError(f"Axis detection failed: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error in axis detection: {e}", exc_info=True)
            raise ChartDetectionError(f"Axis detection failed: {e}")

    def _find_bottom_line_validated(
        self, h_lines: List[np.ndarray], img_height: int, img_width: int
    ) -> Optional[np.ndarray]:
        """
        Find X-axis (bottom horizontal line) with validation.

        Validates that the line is:
        - In the bottom 30% of the image
        - Long enough (>50% of image width)
        - Nearly horizontal (<5 degrees)

        Args:
            h_lines: List of horizontal lines
            img_height: Image height
            img_width: Image width

        Returns:
            Best X-axis line or None if no valid line found
        """
        if not h_lines:
            return None

        candidates = []
        min_y = self.config.axis_position_thresholds["x_axis_min_y_ratio"] * img_height
        min_length = self.config.axis_length_threshold_ratio * img_width

        for line in h_lines:
            x1, y1, x2, y2 = line
            y_pos = (y1 + y2) / 2
            length = abs(x2 - x1)

            # Validation 1: Must be in bottom 30% of image
            if y_pos < min_y:
                continue

            # Validation 2: Must be long enough
            if length < min_length:
                continue

            # Validation 3: Must be nearly horizontal
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 5:
                continue

            # Score based on position and length
            # Prefer lines lower in image and longer
            position_score = y_pos / img_height
            length_score = length / img_width
            total_score = position_score * 0.6 + length_score * 0.4

            candidates.append(
                {"line": line, "y_pos": y_pos, "length": length, "score": total_score}
            )

        if not candidates:
            self.logger.debug(
                f"No valid X-axis candidates found among {len(h_lines)} horizontal lines"
            )
            return None

        # Return best candidate
        best = max(candidates, key=lambda c: c["score"])
        self.logger.debug(
            f"Found X-axis at y={best['y_pos']:.1f}, length={best['length']:.1f}, "
            f"score={best['score']:.3f}"
        )
        return best["line"]

    def _find_left_line_validated(
        self, v_lines: List[np.ndarray], img_height: int, img_width: int
    ) -> Optional[np.ndarray]:
        """
        Find Y-axis (left vertical line) with validation.

        Validates that the line is:
        - In the left 20% of the image
        - Long enough (>50% of image height)
        - Nearly vertical (<5 degrees from vertical)

        Args:
            v_lines: List of vertical lines
            img_height: Image height
            img_width: Image width

        Returns:
            Best Y-axis line or None if no valid line found
        """
        if not v_lines:
            return None

        candidates = []
        max_x = self.config.axis_position_thresholds["y_axis_max_x_ratio"] * img_width
        min_length = self.config.axis_length_threshold_ratio * img_height

        for line in v_lines:
            x1, y1, x2, y2 = line
            x_pos = (x1 + x2) / 2
            length = abs(y2 - y1)

            # Validation 1: Must be in left 20% of image
            if x_pos > max_x:
                continue

            # Validation 2: Must be long enough
            if length < min_length:
                continue

            # Validation 3: Must be nearly vertical
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 85:  # Less than 85 degrees from horizontal (>5 from vertical)
                continue

            # Score based on position and length
            # Prefer lines further left and longer
            position_score = 1 - (x_pos / img_width)
            length_score = length / img_height
            total_score = position_score * 0.6 + length_score * 0.4

            candidates.append(
                {"line": line, "x_pos": x_pos, "length": length, "score": total_score}
            )

        if not candidates:
            self.logger.debug(
                f"No valid Y-axis candidates found among {len(v_lines)} vertical lines"
            )
            return None

        # Return best candidate
        best = max(candidates, key=lambda c: c["score"])
        self.logger.debug(
            f"Found Y-axis at x={best['x_pos']:.1f}, length={best['length']:.1f}, "
            f"score={best['score']:.3f}"
        )
        return best["line"]

    def _fallback_axes(
        self, img_height: int, img_width: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Provide fallback axes when none detected.

        Uses image boundaries as approximate axes.

        Args:
            img_height: Image height
            img_width: Image width

        Returns:
            Tuple of (x_axis, y_axis) using image boundaries
        """
        self.logger.warning("Using fallback axes (image boundaries)")

        # X-axis: bottom edge
        x_axis = np.array([0, img_height - 1, img_width - 1, img_height - 1])

        # Y-axis: left edge
        y_axis = np.array([0, 0, 0, img_height - 1])

        return x_axis, y_axis

    def _merge_nearby_bars_optimized(
        self, bars: List[Dict], max_distance: Optional[int] = None
    ) -> List[Dict]:
        """
        Merge nearby bars using KDTree for O(n log n) performance.

        This replaces the O(n²) nested loop with spatial indexing.

        Args:
            bars: List of bar dictionaries
            max_distance: Maximum distance for merging (uses config if None)

        Returns:
            List of merged bars
        """
        if not bars or len(bars) < 2:
            return bars

        if max_distance is None:
            max_distance = self.config.merge_max_distance

        self.logger.debug(f"Merging {len(bars)} bars with max_distance={max_distance}")

        try:
            # Build KDTree for efficient spatial queries
            centers = np.array([b["center"] for b in bars])
            tree = KDTree(centers)

            merged_bars = []
            used_indices = set()

            for i, bar in enumerate(bars):
                if i in used_indices:
                    continue

                # Query nearby bars efficiently - O(log n)
                indices = tree.query_ball_point(bar["center"], r=max_distance)

                # Collect bars to merge
                bars_to_merge = []
                for idx in indices:
                    if idx not in used_indices:
                        # Additional validation: check vertical overlap
                        x1, y1, w1, h1 = bar["bbox"]
                        x2, y2, w2, h2 = bars[idx]["bbox"]
                        vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

                        if vertical_overlap > 0.5 * min(h1, h2):
                            bars_to_merge.append(bars[idx])
                            used_indices.add(idx)

                # Merge if multiple bars found
                if len(bars_to_merge) == 1:
                    merged_bars.append(bar)
                else:
                    merged_bar = self._create_merged_bar(bars_to_merge)
                    merged_bars.append(merged_bar)

            self.logger.debug(f"Merged into {len(merged_bars)} bars")
            return merged_bars

        except Exception as e:
            self.logger.warning(
                f"KDTree merge failed: {e}. Falling back to original method."
            )
            return self._merge_nearby_bars_fallback(bars, max_distance)

    def _create_merged_bar(self, bars_to_merge: List[Dict]) -> Dict:
        """
        Create a merged bar from multiple bars.

        Args:
            bars_to_merge: List of bars to merge

        Returns:
            Merged bar dictionary
        """
        all_x = [b["bbox"][0] for b in bars_to_merge]
        all_y = [b["bbox"][1] for b in bars_to_merge]
        all_x2 = [b["bbox"][0] + b["bbox"][2] for b in bars_to_merge]
        all_y2 = [b["bbox"][1] + b["bbox"][3] for b in bars_to_merge]

        merged_x = min(all_x)
        merged_y = min(all_y)
        merged_w = max(all_x2) - merged_x
        merged_h = max(all_y2) - merged_y

        return {
            "bbox": (merged_x, merged_y, merged_w, merged_h),
            "area": merged_w * merged_h,
            "center": (merged_x + merged_w // 2, merged_y + merged_h // 2),
        }

    def _merge_nearby_bars_fallback(
        self, bars: List[Dict], max_distance: int
    ) -> List[Dict]:
        """
        Fallback O(n²) merge method (original implementation).

        Used if KDTree method fails.
        """
        if not bars:
            return bars

        merged_bars = []
        used_indices = set()

        for i, bar1 in enumerate(bars):
            if i in used_indices:
                continue

            x1, y1, w1, h1 = bar1["bbox"]
            bars_to_merge = [bar1]
            used_indices.add(i)

            for j, bar2 in enumerate(bars):
                if j <= i or j in used_indices:
                    continue

                x2, y2, w2, h2 = bar2["bbox"]

                horizontal_gap = abs((x1 + w1 / 2) - (x2 + w2 / 2))
                vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

                if horizontal_gap < max_distance and vertical_overlap > 0.5 * min(
                    h1, h2
                ):
                    bars_to_merge.append(bar2)
                    used_indices.add(j)

            if len(bars_to_merge) == 1:
                merged_bars.append(bar1)
            else:
                merged_bars.append(self._create_merged_bar(bars_to_merge))

        return merged_bars

    def detect_bars(
        self, image, x_axis: Optional[np.ndarray], y_axis: Optional[np.ndarray]
    ) -> List[Dict]:
        """
        Detect bars using connected components and morphological operations.

        Uses multi-scale detection, duplicate removal, merging, and validation.

        Args:
            image: RGB image numpy array
            x_axis: X-axis line [x1, y1, x2, y2] or None
            y_axis: Y-axis line [x1, y1, x2, y2] or None

        Returns:
            List of bar dictionaries with 'bbox', 'area', 'center' keys

        Raises:
            InvalidImageError: If image is invalid
            BarDetectionError: If detection fails critically
        """
        # Validate input
        self._validate_image(image)

        h, w = image.shape[:2]
        self.logger.info(f"Detecting bars for image size: {h}x{w}")

        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Create mask for colored regions (non-white)
            lower_white = np.array(self.config.hsv_lower_white)
            upper_white = np.array(self.config.hsv_upper_white)
            mask = cv2.bitwise_not(cv2.inRange(hsv, lower_white, upper_white))

            # Multi-scale detection
            bars_all = []
            x_axis_y = x_axis[1] if x_axis is not None else h - 50

            self.logger.debug(f"Using kernel sizes: {self.config.kernel_sizes}")

            for kernel_size in self.config.kernel_sizes:
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                eroded = cv2.erode(
                    mask, kernel_erode, iterations=self.config.erode_iterations
                )

                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                processed_mask = cv2.dilate(
                    eroded, kernel_dilate, iterations=self.config.dilate_iterations
                )

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    processed_mask, connectivity=8
                )

                # Use adaptive thresholds from config
                min_area = self.config.min_bar_area
                min_aspect_ratio = self.config.min_aspect_ratio
                max_aspect_ratio = self.config.max_aspect_ratio

                for i in range(1, num_labels):  # Skip label 0 (background)
                    area = stats[i, cv2.CC_STAT_AREA]
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w_bar = stats[i, cv2.CC_STAT_WIDTH]
                    h_bar = stats[i, cv2.CC_STAT_HEIGHT]

                    aspect_ratio = h_bar / w_bar if w_bar > 0 else 0

                    # Bar must be above x-axis and have reasonable size
                    if (
                        area > min_area
                        and min_aspect_ratio < aspect_ratio < max_aspect_ratio
                        and y < x_axis_y
                        and h_bar > self.config.min_bar_height
                    ):
                        bars_all.append(
                            {
                                "bbox": (x, y, w_bar, h_bar),
                                "area": area,
                                "center": (int(centroids[i][0]), int(centroids[i][1])),
                            }
                        )

            self.logger.debug(f"Detected {len(bars_all)} bars before filtering")

            # Sort by x position
            bars_all = sorted(bars_all, key=lambda b: b["bbox"][0])

            # Remove duplicates
            bars = self._remove_duplicates(bars_all)
            self.logger.debug(f"After duplicate removal: {len(bars)} bars")

            # Sort again after duplicate removal
            bars = sorted(bars, key=lambda b: b["bbox"][0])

            # Merge nearby bars
            bars = self._merge_nearby_bars_optimized(bars)
            self.logger.debug(f"After merging: {len(bars)} bars")

            # Validate bars using pipeline
            bars = self.validation_pipeline.validate(bars)
            self.logger.info(f"Final bar count after validation: {len(bars)}")

            return bars

        except cv2.error as e:
            self.logger.error(f"OpenCV error in bar detection: {e}")
            raise BarDetectionError(f"Bar detection failed: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error in bar detection: {e}", exc_info=True)
            raise BarDetectionError(f"Bar detection failed: {e}")

    def _remove_duplicates(self, bars: List[Dict]) -> List[Dict]:
        """
        Remove duplicate bars based on overlap.

        Args:
            bars: List of bars (sorted by x position)

        Returns:
            List of unique bars
        """
        if not bars:
            return bars

        unique_bars = []
        overlap_threshold = self.config.overlap_threshold

        for bar in bars:
            is_duplicate = False
            x1, y1, w1, h1 = bar["bbox"]

            for existing_bar in unique_bars:
                x2, y2, w2, h2 = existing_bar["bbox"]

                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                bar_area = w1 * h1
                existing_area = w2 * h2

                # Check if overlap exceeds threshold
                if overlap_area > overlap_threshold * min(bar_area, existing_area):
                    is_duplicate = True
                    # Keep bar with larger area
                    if bar_area > existing_area:
                        unique_bars.remove(existing_bar)
                        unique_bars.append(bar)
                    break

            if not is_duplicate:
                unique_bars.append(bar)

        return unique_bars

    def extract_bar_colors(self, image, bars: List[Dict]) -> List[Tuple[int, int, int]]:
        """
        Extract average color for each bar.

        Args:
            image: RGB image numpy array
            bars: List of bar dictionaries

        Returns:
            List of RGB color tuples
        """
        colors = []

        for bar in bars:
            x, y, w, h = bar["bbox"]

            # Extract bar region
            bar_region = image[y : y + h, x : x + w]

            # Calculate average color
            avg_color = np.mean(bar_region, axis=(0, 1)).astype(int)
            colors.append(tuple(avg_color))

        return colors

    def detect_text_regions(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using morphological operations.

        Args:
            image: RGB image numpy array

        Returns:
            List of (x, y, w, h) tuples for text regions
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Threshold
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Dilate to connect text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            text_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Filter based on aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 20 and w > 10 and h > 10:
                    text_regions.append((x, y, w, h))

            self.logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions

        except Exception as e:
            self.logger.error(f"Error detecting text regions: {e}")
            return []
