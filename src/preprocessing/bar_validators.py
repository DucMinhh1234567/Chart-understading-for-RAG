# src/preprocessing/bar_validators.py
"""
Validation pipeline for bar detection using Chain of Responsibility pattern.

Each validator is independent, testable, and can be composed into a pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


logger = logging.getLogger(__name__)


# ==================== BASE VALIDATOR ====================


class BarValidator(ABC):
    """
    Abstract base class for bar validators.

    Each validator implements a single validation concern and can be
    chained together in a ValidationPipeline.
    """

    @abstractmethod
    def validate(self, bars: List[Dict]) -> List[Dict]:
        """
        Validate and filter bars.

        Args:
            bars: List of bar dictionaries with 'bbox', 'area', 'center' keys

        Returns:
            Filtered list of bars that pass validation
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return validator name for logging"""
        pass


# ==================== INDIVIDUAL VALIDATORS ====================


class WidthValidator(BarValidator):
    """
    Validates bars based on width consistency.

    Real bars in a chart have consistent widths. This validator removes
    bars with widths that deviate significantly from the median.

    This is SAFE for data outliers because width is independent of value.
    """

    def __init__(self, min_ratio: float = 0.35, max_ratio: float = 2.2):
        """
        Args:
            min_ratio: Minimum width as ratio of median (default 35%)
            max_ratio: Maximum width as ratio of median (default 220%)
        """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def get_name(self) -> str:
        return "WidthValidator"

    def validate(self, bars: List[Dict]) -> List[Dict]:
        if len(bars) < 2:
            logger.debug(f"{self.get_name()}: Too few bars ({len(bars)}), skipping")
            return bars

        # Calculate median width
        widths = [b["bbox"][2] for b in bars]
        median_width = np.median(widths)

        # Filter bars
        min_width = self.min_ratio * median_width
        max_width = self.max_ratio * median_width

        filtered_bars = [b for b in bars if min_width <= b["bbox"][2] <= max_width]

        removed = len(bars) - len(filtered_bars)
        if removed > 0:
            logger.debug(
                f"{self.get_name()}: Removed {removed} bars. "
                f"Median width: {median_width:.1f}, "
                f"Valid range: [{min_width:.1f}, {max_width:.1f}]"
            )

        return filtered_bars


class AreaValidator(BarValidator):
    """
    Validates bars based on minimum area.

    Removes tiny noise regions (grid lines, artifacts) while keeping
    legitimate small bars. Uses dynamic threshold based on median area.
    """

    def __init__(self, min_ratio: float = 0.12, absolute_min: int = 150):
        """
        Args:
            min_ratio: Minimum area as ratio of median (default 12%)
            absolute_min: Absolute minimum area in pixels (default 150)
        """
        self.min_ratio = min_ratio
        self.absolute_min = absolute_min

    def get_name(self) -> str:
        return "AreaValidator"

    def validate(self, bars: List[Dict]) -> List[Dict]:
        if len(bars) < 2:
            logger.debug(f"{self.get_name()}: Too few bars ({len(bars)}), skipping")
            return bars

        # Calculate median area
        areas = [b["area"] for b in bars]
        median_area = np.median(areas)

        # Use the larger of: absolute minimum or ratio-based threshold
        min_area_threshold = max(self.absolute_min, self.min_ratio * median_area)

        # Filter bars
        filtered_bars = [b for b in bars if b["area"] >= min_area_threshold]

        removed = len(bars) - len(filtered_bars)
        if removed > 0:
            logger.debug(
                f"{self.get_name()}: Removed {removed} bars. "
                f"Median area: {median_area:.1f}, "
                f"Min threshold: {min_area_threshold:.1f}"
            )

        return filtered_bars


class AspectRatioValidator(BarValidator):
    """
    Validates bars based on aspect ratio (height/width).

    Bars should be somewhat vertical (for vertical bar charts).
    Filters out grid lines (very flat) and noise (very thin).
    """

    def __init__(self, min_ratio: float = 0.4, max_ratio: float = 15):
        """
        Args:
            min_ratio: Minimum height/width ratio (default 0.4)
            max_ratio: Maximum height/width ratio (default 15)
        """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def get_name(self) -> str:
        return "AspectRatioValidator"

    def validate(self, bars: List[Dict]) -> List[Dict]:
        if len(bars) < 2:
            logger.debug(f"{self.get_name()}: Too few bars ({len(bars)}), skipping")
            return bars

        filtered_bars = []
        for b in bars:
            width = b["bbox"][2]
            height = b["bbox"][3]
            aspect_ratio = height / width if width > 0 else 0

            if self.min_ratio <= aspect_ratio <= self.max_ratio:
                filtered_bars.append(b)

        removed = len(bars) - len(filtered_bars)
        if removed > 0:
            logger.debug(
                f"{self.get_name()}: Removed {removed} bars. "
                f"Valid aspect ratio range: [{self.min_ratio}, {self.max_ratio}]"
            )

        return filtered_bars


class SpacingValidator(BarValidator):
    """
    Validates bars based on horizontal spacing consistency.

    Real bars in charts are typically evenly spaced. This validator
    identifies and removes bars that break the spacing pattern.

    This is the MOST POWERFUL validator for distinguishing real bars
    from false positives.
    """

    def __init__(self, tolerance: float = 0.4):
        """
        Args:
            tolerance: Maximum spacing variation as ratio of mean (default 40%)
        """
        self.tolerance = tolerance

    def get_name(self) -> str:
        return "SpacingValidator"

    def validate(self, bars: List[Dict]) -> List[Dict]:
        if len(bars) < 3:
            logger.debug(f"{self.get_name()}: Too few bars ({len(bars)}), skipping")
            return bars

        # Sort bars by x position
        sorted_bars = sorted(bars, key=lambda b: b["center"][0])

        # Calculate spacings between consecutive bars
        spacings = []
        for i in range(len(sorted_bars) - 1):
            spacing = sorted_bars[i + 1]["center"][0] - sorted_bars[i]["center"][0]
            spacings.append(spacing)

        if len(spacings) < 2:
            return bars

        # Calculate statistics
        avg_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)

        # Check if spacing is inconsistent
        if std_spacing <= self.tolerance * avg_spacing:
            # Spacing is consistent, keep all bars
            logger.debug(
                f"{self.get_name()}: Spacing is consistent. "
                f"Avg: {avg_spacing:.1f}, Std: {std_spacing:.1f}"
            )
            return bars

        # Spacing is inconsistent - remove outliers
        logger.debug(
            f"{self.get_name()}: Spacing is inconsistent. "
            f"Avg: {avg_spacing:.1f}, Std: {std_spacing:.1f}. "
            f"Filtering outliers..."
        )

        # Keep first bar, then validate others based on spacing
        validated_bars = [sorted_bars[0]]

        # Get median width for additional validation
        widths = [b["bbox"][2] for b in sorted_bars]
        median_width = np.median(widths)

        for i in range(1, len(sorted_bars)):
            spacing_before = (
                sorted_bars[i]["center"][0] - sorted_bars[i - 1]["center"][0]
            )

            # Check if spacing is close to average
            spacing_deviation = abs(spacing_before - avg_spacing) / avg_spacing

            if spacing_deviation < 0.5:  # Within 50% of average
                validated_bars.append(sorted_bars[i])
            else:
                # Spacing is unusual - check width as secondary validation
                width_deviation = (
                    abs(sorted_bars[i]["bbox"][2] - median_width) / median_width
                )

                if width_deviation < 0.4:  # Width is normal
                    # Might be a real bar with unusual spacing (e.g., gap in data)
                    validated_bars.append(sorted_bars[i])
                else:
                    # Both spacing and width are unusual - likely false positive
                    logger.debug(
                        f"{self.get_name()}: Removed bar at x={sorted_bars[i]['bbox'][0]} "
                        f"(spacing deviation: {spacing_deviation:.2f}, "
                        f"width deviation: {width_deviation:.2f})"
                    )

        removed = len(bars) - len(validated_bars)
        if removed > 0:
            logger.debug(f"{self.get_name()}: Removed {removed} bars based on spacing")

        return validated_bars


# ==================== VALIDATION PIPELINE ====================


class ValidationPipeline:
    """
    Chains multiple validators together.

    Validators are applied in sequence with early exit if too few bars remain.
    Each step is logged for debugging.
    """

    def __init__(self, validators: List[BarValidator]):
        """
        Args:
            validators: List of validators to apply in sequence
        """
        self.validators = validators
        self.logger = logging.getLogger(__name__)

    def validate(self, bars: List[Dict]) -> List[Dict]:
        """
        Apply all validators in sequence.

        Args:
            bars: List of detected bars

        Returns:
            Filtered list of bars that pass all validations
        """
        if not bars:
            self.logger.debug("ValidationPipeline: No bars to validate")
            return bars

        initial_count = len(bars)
        self.logger.debug(f"ValidationPipeline: Starting with {initial_count} bars")

        current_bars = bars

        for i, validator in enumerate(self.validators, 1):
            before_count = len(current_bars)
            current_bars = validator.validate(current_bars)
            after_count = len(current_bars)

            self.logger.debug(
                f"ValidationPipeline: Step {i}/{len(self.validators)} - "
                f"{validator.get_name()}: {before_count} -> {after_count} bars"
            )

            # Early exit if too few bars remain
            if len(current_bars) < 2:
                self.logger.debug(
                    f"ValidationPipeline: Early exit - only {len(current_bars)} bars remaining"
                )
                break

        final_count = len(current_bars)
        removed = initial_count - final_count

        if removed > 0:
            self.logger.info(
                f"ValidationPipeline: Filtered {removed}/{initial_count} bars. "
                f"Remaining: {final_count}"
            )

        return current_bars

    def add_validator(self, validator: BarValidator) -> None:
        """Add a validator to the pipeline"""
        self.validators.append(validator)

    def remove_validator(self, validator_name: str) -> bool:
        """
        Remove a validator by name.

        Returns:
            True if validator was found and removed, False otherwise
        """
        for i, validator in enumerate(self.validators):
            if validator.get_name() == validator_name:
                self.validators.pop(i)
                return True
        return False
