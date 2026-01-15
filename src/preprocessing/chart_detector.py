"""Chart component detection for chart-understanding.

This module provides :class:`ChartComponentDetector` for detecting axes, bars,
colors, and text regions in chart images using computer vision techniques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.preprocessing.image_utils import ImagePreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class ChartComponentDetector:
    """Detects chart components (axes, bars, colors, text) from preprocessed images.

    This class uses a combination of techniques:
    - Hough Line Transform for axis detection
    - HSV color space + contours for bar detection
    - Morphological operations for text region detection

    Attributes
    ----------
    preprocessor: ImagePreprocessor
        Instance used for preprocessing steps if needed.
    """

    def __init__(self, preprocessor: ImagePreprocessor | None = None) -> None:
        """Initialize the detector.

        Parameters
        ----------
        preprocessor:
            Optional ImagePreprocessor instance. If None, creates a new one.
        """
        self.preprocessor = preprocessor or ImagePreprocessor()

    def detect_axes(
        self, image: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Detect X-axis (bottom horizontal) and Y-axis (left vertical) using Hough Line Transform.

        Steps:
        1. Convert to grayscale and detect edges
        2. Apply Hough Line Transform
        3. Classify lines as horizontal or vertical
        4. Find bottom-most horizontal line (X-axis)
        5. Find left-most vertical line (Y-axis)

        Parameters
        ----------
        image:
            Preprocessed RGB image (H, W, 3) or grayscale (H, W).

        Returns
        -------
        tuple[tuple | None, tuple | None]
            (x_axis, y_axis) where each axis is (x1, y1, x2, y2) coordinates,
            or None if not found.

        Notes
        -----
        Uses Probabilistic Hough Line Transform (HoughLinesP) for better
        performance and more accurate line segments.
        """
        try:
            # Convert to grayscale if needed
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Detect edges using Canny
            edges = self.preprocessor.detect_edges(gray)

            # Apply Hough Line Transform (Probabilistic for better results)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=50,
                maxLineGap=10,
            )

            if lines is None or len(lines) == 0:
                return None, None

            # Get image dimensions
            h, w = gray.shape[:2]

            # Classify lines into horizontal and vertical
            horizontal_lines: list[tuple[int, int, int, int]] = []
            vertical_lines: list[tuple[int, int, int, int]] = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate angle (in degrees)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Classify: horizontal if angle < 15 or > 165, vertical if 75 < angle < 105
                if angle < 15 or angle > 165:
                    horizontal_lines.append((x1, y1, x2, y2))
                elif 75 < angle < 105:
                    vertical_lines.append((x1, y1, x2, y2))

            # Find X-axis (bottom horizontal line)
            x_axis = self._find_bottom_line(horizontal_lines, h)

            # Find Y-axis (left vertical line)
            y_axis = self._find_left_line(vertical_lines, w)

            return x_axis, y_axis

        except Exception as exc:
            raise RuntimeError(f"Failed to detect axes: {exc}") from exc

    def detect_bars(
        self,
        image: np.ndarray,
        x_axis: tuple[int, int, int, int] | None = None,
        y_axis: tuple[int, int, int, int] | None = None,
        min_area: int = 500,
        min_aspect_ratio: float = 0.5,
    ) -> list[dict]:
        """Detect bar chart bars using HSV color space and contour analysis.

        Steps:
        1. Convert RGB to HSV
        2. Create mask to remove white/light background
        3. Find contours
        4. Filter contours by area, aspect ratio, and position
        5. Sort bars left to right

        Parameters
        ----------
        image:
            Preprocessed RGB image (H, W, 3).
        x_axis:
            Optional X-axis coordinates (x1, y1, x2, y2). If provided, bars
            must be above this line.
        y_axis:
            Optional Y-axis coordinates (x1, y1, x2, y2). Used for reference.
        min_area:
            Minimum contour area in pixels to be considered a bar.
        min_aspect_ratio:
            Minimum height/width ratio. Bars should be taller than wide.

        Returns
        -------
        list[dict]
            List of bar dictionaries, each containing:
            - 'bbox': (x, y, w, h) bounding box
            - 'contour': contour array
            - 'area': contour area (float)
            - 'center': (cx, cy) centroid coordinates
            Sorted left to right by x-coordinate.
        """
        try:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("detect_bars expects an RGB image with shape (H, W, 3).")

            # Convert to HSV for better color-based masking
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Create mask to remove white/light background
            # Lower and upper bounds for white/light colors in HSV
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            mask = cv2.inRange(hsv, lower_white, upper_white)
            mask_inv = cv2.bitwise_not(mask)

            # Also remove very dark regions (axes, grid lines)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 50])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            mask_inv = cv2.bitwise_and(mask_inv, cv2.bitwise_not(mask_dark))

            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
            mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return []

            bars: list[dict] = []

            # Get X-axis Y coordinate for filtering
            x_axis_y: int | None = None
            if x_axis is not None:
                # Use the Y coordinate from axis (assuming it's roughly horizontal)
                x_axis_y = max(x_axis[1], x_axis[3])

            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)

                if area < min_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (height should be greater than width for bars)
                if h == 0:
                    continue
                aspect_ratio = h / max(w, 1)

                if aspect_ratio < min_aspect_ratio:
                    continue

                # Filter by position: bars should be above X-axis
                if x_axis_y is not None:
                    bar_bottom = y + h
                    # Allow small tolerance (10px) in case of slight misalignment
                    if bar_bottom > x_axis_y + 10:
                        continue

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                bars.append(
                    {
                        "bbox": (x, y, w, h),
                        "contour": contour,
                        "area": float(area),
                        "center": (cx, cy),
                    }
                )

            # Sort bars left to right by x-coordinate
            bars.sort(key=lambda b: b["bbox"][0])

            return bars

        except Exception as exc:
            raise RuntimeError(f"Failed to detect bars: {exc}") from exc

    def extract_bar_colors(self, image: np.ndarray, bars: Sequence[dict]) -> list[tuple[int, int, int]]:
        """Extract average RGB color for each detected bar.

        Parameters
        ----------
        image:
            Original RGB image (H, W, 3).
        bars:
            List of bar dictionaries from detect_bars().

        Returns
        -------
        list[tuple[int, int, int]]
            List of RGB color tuples, one per bar, in the same order as input.
        """
        try:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("extract_bar_colors expects an RGB image with shape (H, W, 3).")

            colors: list[tuple[int, int, int]] = []

            for bar in bars:
                x, y, w, h = bar["bbox"]

                # Extract region of interest (ROI)
                roi = image[y : y + h, x : x + w]

                # Calculate mean color (excluding background/transparent pixels)
                # Create mask for non-white pixels to avoid background contamination
                mask = cv2.inRange(roi, (0, 0, 0), (250, 250, 250))
                mask_inv = cv2.bitwise_not(mask)

                if np.sum(mask_inv) == 0:
                    # If all pixels are white/background, use mean of entire ROI
                    mean_color = roi.mean(axis=(0, 1))
                else:
                    # Calculate mean only for non-background pixels
                    mean_color = roi[mask_inv > 0].mean(axis=0)

                # Convert to integers and ensure valid RGB range
                r = int(np.clip(mean_color[0], 0, 255))
                g = int(np.clip(mean_color[1], 0, 255))
                b = int(np.clip(mean_color[2], 0, 255))

                colors.append((r, g, b))

            return colors

        except Exception as exc:
            raise RuntimeError(f"Failed to extract bar colors: {exc}") from exc

    def detect_text_regions(
        self,
        image: np.ndarray,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 20.0,
    ) -> list[tuple[int, int, int, int]]:
        """Detect text regions using morphological operations and contour analysis.

        Steps:
        1. Convert to grayscale and binarize
        2. Apply morphological dilation to connect text characters
        3. Find contours
        4. Filter by aspect ratio (text regions are typically rectangular)

        Parameters
        ----------
        image:
            Preprocessed RGB image (H, W, 3) or grayscale (H, W).
        min_aspect_ratio:
            Minimum width/height or height/width ratio for text regions.
        max_aspect_ratio:
            Maximum width/height or height/width ratio for text regions.

        Returns
        -------
        list[tuple[int, int, int, int]]
            List of bounding boxes (x, y, w, h) for detected text regions.
        """
        try:
            # Convert to grayscale if needed
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Binarize image using Otsu's method
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Apply morphological dilation to connect text characters into regions
            # Horizontal kernel for connecting characters in words
            kernel_h = np.ones((1, 15), np.uint8)
            dilated = cv2.dilate(binary, kernel_h, iterations=2)

            # Also try vertical kernel for vertical text
            kernel_v = np.ones((15, 1), np.uint8)
            dilated = cv2.dilate(dilated, kernel_v, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return []

            text_regions: list[tuple[int, int, int, int]] = []

            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio
                if w == 0 or h == 0:
                    continue

                # Calculate aspect ratio (both orientations)
                aspect_h = w / h  # horizontal text
                aspect_v = h / w  # vertical text

                # Check if aspect ratio is within acceptable range
                if (
                    min_aspect_ratio <= aspect_h <= max_aspect_ratio
                    or min_aspect_ratio <= aspect_v <= max_aspect_ratio
                ):
                    text_regions.append((x, y, w, h))

            # Sort by position (top to bottom, left to right)
            text_regions.sort(key=lambda bbox: (bbox[1], bbox[0]))

            return text_regions

        except Exception as exc:
            raise RuntimeError(f"Failed to detect text regions: {exc}") from exc

    def _find_bottom_line(
        self, horizontal_lines: Sequence[tuple[int, int, int, int]], img_height: int
    ) -> tuple[int, int, int, int] | None:
        """Find the bottom-most horizontal line (X-axis candidate).

        Parameters
        ----------
        horizontal_lines:
            List of horizontal line coordinates (x1, y1, x2, y2).
        img_height:
            Image height in pixels.

        Returns
        -------
        tuple[int, int, int, int] | None
            Bottom-most line coordinates (x1, y1, x2, y2), or None if no lines.
        """
        if len(horizontal_lines) == 0:
            return None

        # Find line with maximum Y coordinate (bottom-most)
        # Use the average Y of both endpoints
        bottom_line = max(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)

        # Only consider lines in the bottom portion of image (lower 40%)
        bottom_threshold = img_height * 0.6
        line_y_avg = (bottom_line[1] + bottom_line[3]) / 2

        if line_y_avg < bottom_threshold:
            return None

        return bottom_line

    def _find_left_line(
        self, vertical_lines: Sequence[tuple[int, int, int, int]], img_width: int
    ) -> tuple[int, int, int, int] | None:
        """Find the left-most vertical line (Y-axis candidate).

        Parameters
        ----------
        vertical_lines:
            List of vertical line coordinates (x1, y1, x2, y2).
        img_width:
            Image width in pixels.

        Returns
        -------
        tuple[int, int, int, int] | None
            Left-most line coordinates (x1, y1, x2, y2), or None if no lines.
        """
        if len(vertical_lines) == 0:
            return None

        # Find line with minimum X coordinate (left-most)
        # Use the average X of both endpoints
        left_line = min(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)

        # Only consider lines in the left portion of image (left 40%)
        left_threshold = img_width * 0.4
        line_x_avg = (left_line[0] + left_line[2]) / 2

        if line_x_avg > left_threshold:
            return None
        return left_line
