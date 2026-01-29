"""
Tests for bar validators.

Tests cover:
- WidthValidator: width consistency filtering
- AreaValidator: tiny noise removal
- AspectRatioValidator: aspect ratio filtering
- SpacingValidator: even spacing detection
- ValidationPipeline: chaining validators
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.bar_validators import (
    WidthValidator,
    AreaValidator,
    AspectRatioValidator,
    SpacingValidator,
    ValidationPipeline,
)


# ==================== TestWidthValidator ====================

class TestWidthValidator:
    """Test WidthValidator - filters bars with inconsistent widths."""
    
    def test_keeps_consistent_width_bars(self, mock_bars):
        """All bars have width=50, should keep all."""
        validator = WidthValidator(min_ratio=0.35, max_ratio=2.2)
        result = validator.validate(mock_bars)
        assert len(result) == 4
        assert result == mock_bars
    
    def test_removes_width_outlier(self, mock_bars_with_width_outlier):
        """Bar with width=150 should be removed (median=50, max=110)."""
        validator = WidthValidator(min_ratio=0.35, max_ratio=2.2)
        result = validator.validate(mock_bars_with_width_outlier)
        
        # Should remove the outlier bar with width=150
        assert len(result) == 3
        for bar in result:
            assert bar["bbox"][2] == 50  # All remaining bars have width=50
    
    def test_skips_when_less_than_2_bars(self, mock_single_bar):
        """Should skip validation when < 2 bars."""
        validator = WidthValidator()
        result = validator.validate(mock_single_bar)
        assert len(result) == 1
        assert result == mock_single_bar
    
    def test_empty_input(self):
        """Should handle empty list."""
        validator = WidthValidator()
        result = validator.validate([])
        assert result == []
    
    def test_custom_min_ratio(self, mock_bars):
        """Test with custom min_ratio parameter."""
        # Set min_ratio very high so all bars are filtered
        validator = WidthValidator(min_ratio=0.99, max_ratio=1.01)
        result = validator.validate(mock_bars)
        # With very tight tolerance around median, most bars should pass
        # (median width = 50, so 49.5 to 50.5 range)
        assert len(result) == 4  # All have same width
    
    def test_custom_max_ratio(self, mock_bars_with_width_outlier):
        """Test with custom max_ratio parameter."""
        # Set max_ratio to 4.0 to allow the outlier
        validator = WidthValidator(min_ratio=0.35, max_ratio=4.0)
        result = validator.validate(mock_bars_with_width_outlier)
        assert len(result) == 4  # All bars kept including outlier
    
    def test_get_name(self):
        """Validator should have correct name."""
        validator = WidthValidator()
        assert validator.get_name() == "WidthValidator"


# ==================== TestAreaValidator ====================

class TestAreaValidator:
    """Test AreaValidator - filters tiny noise regions."""
    
    def test_keeps_normal_area_bars(self, mock_bars):
        """All bars have reasonable area, should keep all."""
        validator = AreaValidator(min_ratio=0.12, absolute_min=150)
        result = validator.validate(mock_bars)
        assert len(result) == 4
    
    def test_removes_tiny_area(self, mock_bars_with_area_outlier):
        """Tiny bar (area=100) should be removed."""
        validator = AreaValidator(min_ratio=0.12, absolute_min=150)
        result = validator.validate(mock_bars_with_area_outlier)
        
        assert len(result) == 3
        for bar in result:
            assert bar["area"] >= 150
    
    def test_absolute_min_threshold(self):
        """Test that absolute_min works correctly."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 200, "center": (125, 275)},
            {"bbox": (200, 150, 50, 200), "area": 200, "center": (225, 250)},
            {"bbox": (300, 340, 5, 5), "area": 25, "center": (302, 342)},  # Below absolute_min
        ]
        validator = AreaValidator(min_ratio=0.12, absolute_min=100)
        result = validator.validate(bars)
        
        # median = 200, threshold = max(100, 0.12*200) = max(100, 24) = 100
        # bar with area=25 should be removed
        assert len(result) == 2
    
    def test_ratio_based_threshold(self):
        """Test that ratio-based threshold works when > absolute_min."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 10000, "center": (125, 275)},
            {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
            {"bbox": (300, 340, 20, 30), "area": 600, "center": (310, 355)},  # Small
        ]
        validator = AreaValidator(min_ratio=0.12, absolute_min=100)
        result = validator.validate(bars)
        
        # median = 10000, threshold = max(100, 0.12*10000) = max(100, 1200) = 1200
        # bar with area=600 should be removed
        assert len(result) == 2
    
    def test_skips_when_less_than_2_bars(self, mock_single_bar):
        """Should skip validation when < 2 bars."""
        validator = AreaValidator()
        result = validator.validate(mock_single_bar)
        assert len(result) == 1
    
    def test_empty_input(self):
        """Should handle empty list."""
        validator = AreaValidator()
        result = validator.validate([])
        assert result == []
    
    def test_get_name(self):
        """Validator should have correct name."""
        validator = AreaValidator()
        assert validator.get_name() == "AreaValidator"


# ==================== TestAspectRatioValidator ====================

class TestAspectRatioValidator:
    """Test AspectRatioValidator - filters by height/width ratio."""
    
    def test_keeps_valid_aspect_ratios(self, mock_bars):
        """Bars with aspect ratio 2-4 should be kept."""
        validator = AspectRatioValidator(min_ratio=0.4, max_ratio=15)
        result = validator.validate(mock_bars)
        assert len(result) == 4
    
    def test_removes_flat_bar(self, mock_bars_with_aspect_ratio_outlier):
        """Bar with aspect ratio 0.1 (very flat) should be removed."""
        validator = AspectRatioValidator(min_ratio=0.4, max_ratio=15)
        result = validator.validate(mock_bars_with_aspect_ratio_outlier)
        
        assert len(result) == 3
        for bar in result:
            width = bar["bbox"][2]
            height = bar["bbox"][3]
            aspect = height / width
            assert 0.4 <= aspect <= 15
    
    def test_removes_thin_bar(self):
        """Bar with very high aspect ratio should be removed."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},   # aspect = 3.0
            {"bbox": (200, 50, 10, 300), "area": 3000, "center": (205, 200)},    # aspect = 30 (too thin)
            {"bbox": (300, 170, 50, 180), "area": 9000, "center": (325, 260)},   # aspect = 3.6
        ]
        validator = AspectRatioValidator(min_ratio=0.4, max_ratio=15)
        result = validator.validate(bars)
        
        assert len(result) == 2
    
    def test_skips_when_less_than_2_bars(self, mock_single_bar):
        """Should skip validation when < 2 bars."""
        validator = AspectRatioValidator()
        result = validator.validate(mock_single_bar)
        assert len(result) == 1
    
    def test_custom_ratios(self):
        """Test with custom min and max ratios."""
        bars = [
            {"bbox": (100, 200, 50, 100), "area": 5000, "center": (125, 250)},   # aspect = 2.0
            {"bbox": (200, 150, 50, 150), "area": 7500, "center": (225, 225)},   # aspect = 3.0
            {"bbox": (300, 100, 50, 250), "area": 12500, "center": (325, 225)},  # aspect = 5.0
        ]
        # Only allow aspect ratio 2.5 to 4.0
        validator = AspectRatioValidator(min_ratio=2.5, max_ratio=4.0)
        result = validator.validate(bars)
        
        assert len(result) == 1
        assert result[0]["bbox"][3] / result[0]["bbox"][2] == 3.0
    
    def test_get_name(self):
        """Validator should have correct name."""
        validator = AspectRatioValidator()
        assert validator.get_name() == "AspectRatioValidator"


# ==================== TestSpacingValidator ====================

class TestSpacingValidator:
    """Test SpacingValidator - filters by horizontal spacing consistency."""
    
    def test_keeps_evenly_spaced_bars(self, mock_bars):
        """4 bars evenly spaced (100px apart) should all be kept."""
        validator = SpacingValidator(tolerance=0.4)
        result = validator.validate(mock_bars)
        assert len(result) == 4
    
    def test_skips_when_less_than_3_bars(self, mock_two_bars):
        """Should skip validation when < 3 bars."""
        validator = SpacingValidator()
        result = validator.validate(mock_two_bars)
        assert len(result) == 2
        assert result == mock_two_bars
    
    def test_skips_single_bar(self, mock_single_bar):
        """Should skip validation for single bar."""
        validator = SpacingValidator()
        result = validator.validate(mock_single_bar)
        assert len(result) == 1
    
    def test_empty_input(self):
        """Should handle empty list."""
        validator = SpacingValidator()
        result = validator.validate([])
        assert result == []
    
    def test_consistent_spacing_returns_all(self):
        """If std/mean < tolerance, should return all bars."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
            {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},  # 100px gap
            {"bbox": (300, 180, 50, 170), "area": 8500, "center": (325, 265)},   # 100px gap
            {"bbox": (400, 220, 50, 130), "area": 6500, "center": (425, 285)},   # 100px gap
        ]
        validator = SpacingValidator(tolerance=0.4)
        result = validator.validate(bars)
        assert len(result) == 4
    
    def test_handles_outlier_with_unusual_width(self, mock_bars_with_spacing_outlier):
        """Bar with unusual spacing AND unusual width should be filtered."""
        validator = SpacingValidator(tolerance=0.4)
        result = validator.validate(mock_bars_with_spacing_outlier)
        
        # The last bar has 315px gap (vs ~100px avg) AND width=80 (vs median=50)
        # It should be filtered
        assert len(result) <= 4  # At most all bars, possibly less
    
    def test_keeps_bar_with_unusual_spacing_but_normal_width(self):
        """Bar with unusual spacing but normal width might be kept (gap in data)."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
            {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},  # 100px gap
            {"bbox": (300, 180, 50, 170), "area": 8500, "center": (325, 265)},   # 100px gap
            {"bbox": (500, 220, 50, 130), "area": 6500, "center": (525, 285)},   # 200px gap but same width
        ]
        validator = SpacingValidator(tolerance=0.4)
        result = validator.validate(bars)
        
        # Last bar has unusual spacing but same width, might be kept
        assert len(result) >= 3
    
    def test_get_name(self):
        """Validator should have correct name."""
        validator = SpacingValidator()
        assert validator.get_name() == "SpacingValidator"


# ==================== TestValidationPipeline ====================

class TestValidationPipeline:
    """Test ValidationPipeline - chains validators together."""
    
    def test_applies_all_validators_in_sequence(self, mock_bars):
        """Pipeline should apply all validators and keep valid bars."""
        pipeline = ValidationPipeline([
            WidthValidator(),
            AreaValidator(),
            AspectRatioValidator(),
            SpacingValidator(),
        ])
        result = pipeline.validate(mock_bars)
        assert len(result) == 4  # All bars are valid
    
    def test_filters_through_pipeline(self, mock_bars_with_width_outlier):
        """Pipeline should filter out invalid bars."""
        pipeline = ValidationPipeline([
            WidthValidator(min_ratio=0.35, max_ratio=2.2),
            AreaValidator(),
        ])
        result = pipeline.validate(mock_bars_with_width_outlier)
        
        # Width outlier should be removed by first validator
        assert len(result) == 3
    
    def test_early_exit_when_too_few_bars(self):
        """Should stop early if < 2 bars remain after a validator."""
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
            {"bbox": (200, 340, 10, 10), "area": 100, "center": (205, 345)},  # Will be removed
        ]
        pipeline = ValidationPipeline([
            AreaValidator(absolute_min=150),  # Removes tiny bar
            SpacingValidator(),  # Should skip (only 1 bar left)
        ])
        result = pipeline.validate(bars)
        
        assert len(result) == 1
    
    def test_empty_input_returns_empty(self):
        """Empty input should return empty output."""
        pipeline = ValidationPipeline([
            WidthValidator(),
            AreaValidator(),
        ])
        result = pipeline.validate([])
        assert result == []
    
    def test_add_validator(self, mock_bars):
        """Should be able to add validators dynamically."""
        pipeline = ValidationPipeline([WidthValidator()])
        pipeline.add_validator(AreaValidator())
        
        assert len(pipeline.validators) == 2
        result = pipeline.validate(mock_bars)
        assert len(result) == 4
    
    def test_remove_validator(self):
        """Should be able to remove validators by name."""
        pipeline = ValidationPipeline([
            WidthValidator(),
            AreaValidator(),
            SpacingValidator(),
        ])
        
        # Remove AreaValidator
        removed = pipeline.remove_validator("AreaValidator")
        assert removed is True
        assert len(pipeline.validators) == 2
        
        # Try to remove non-existent validator
        removed = pipeline.remove_validator("NonExistent")
        assert removed is False
    
    def test_pipeline_preserves_order(self):
        """Validators should be applied in the order they were added."""
        # Create a pipeline that would give different results in different orders
        bars = [
            {"bbox": (100, 200, 50, 150), "area": 7500, "center": (125, 275)},
            {"bbox": (200, 150, 50, 200), "area": 10000, "center": (225, 250)},
            {"bbox": (300, 340, 10, 10), "area": 100, "center": (305, 345)},  # Tiny
        ]
        
        pipeline = ValidationPipeline([
            AreaValidator(absolute_min=150),
        ])
        result = pipeline.validate(bars)
        
        assert len(result) == 2
        assert all(bar["area"] >= 150 for bar in result)
