# src/extraction/bar_extractor.py
"""Bar chart extraction module."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from ..preprocessing.chart_detector import ChartComponentDetector
from ..preprocessing.detector_config import InvalidImageError
from ..config import DEFAULT_LAYOUT
from .ocr_engine import OCREngine


class BarChartExtractor:
    """Extracts structured data from bar chart images."""
    
    def __init__(self) -> None:
        """Initialize extractor with detector and OCR engine."""
        self.detector: ChartComponentDetector = ChartComponentDetector()
        self.ocr: OCREngine = OCREngine()
    
    def _is_number(self, text: str) -> bool:
        """
        Check if text represents a number.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a number, False otherwise
        """
        cleaned = text.replace('.', '').replace(',', '').replace('-', '').replace('%', '').strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            if cleaned and cleaned[0].isdigit():
                return True
            return False
    
    def _extract_ylabel_from_left_region(self, image: np.ndarray) -> Optional[str]:
        """
        Crop left region and OCR to find y-axis label.
        Useful for vertical (rotated 90 degrees) text.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Y-axis label string or None if not found
        """
        h, w = image.shape[:2]
        
        # Crop vùng bên trái
        left_region = image[
            int(DEFAULT_LAYOUT.LEFT_REGION_MIN_Y * h):int(DEFAULT_LAYOUT.LEFT_REGION_MAX_Y * h),
            0:int(DEFAULT_LAYOUT.LEFT_REGION_MAX_X * w)
        ]
        
        # Rotate 90 degrees clockwise để text dọc thành ngang
        rotated = cv2.rotate(left_region, cv2.ROTATE_90_CLOCKWISE)
        
        # OCR với multiple angles
        results = self.ocr.read_text_easyocr(rotated, confidence_threshold=DEFAULT_LAYOUT.YLABEL_CONFIDENCE)
        
        if results:
            # Filter: chỉ lấy text không phải số, dài nhất
            non_number_texts = [r for r in results if not self._is_number(r['text'])]
            if non_number_texts:
                # Sort theo độ dài và confidence
                non_number_texts.sort(key=lambda t: (-len(t['text']), -t['confidence']))
                return non_number_texts[0]['text']
        
        return None
    
    def _load_and_validate_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as RGB numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is unsupported
            InvalidImageError: If image cannot be loaded
        """
        path = Path(image_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Validate image format
        if path.suffix.lower() not in DEFAULT_LAYOUT.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(DEFAULT_LAYOUT.SUPPORTED_IMAGE_FORMATS)}"
            )
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise InvalidImageError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _detect_components(
        self, 
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]], List[Tuple[int, int, int, int]]]:
        """
        Detect chart components (axes, bars, text regions).
        
        Args:
            image: Input image as RGB numpy array
            
        Returns:
            Tuple of (x_axis, y_axis, bars, text_regions)
        """
        x_axis, y_axis = self.detector.detect_axes(image)
        bars = self.detector.detect_bars(image, x_axis, y_axis)
        text_regions = self.detector.detect_text_regions(image)
        
        return x_axis, y_axis, bars, text_regions
    
    def _extract_labels(
        self, 
        image: np.ndarray, 
        text_regions: List[Tuple[int, int, int, int]], 
        ocr_method: str
    ) -> Dict[str, Any]:
        """
        Extract labels from chart using OCR.
        
        Args:
            image: Input image
            text_regions: Detected text regions
            ocr_method: OCR method to use
            
        Returns:
            Labels dict with title, xlabel, ylabel, legend, values
        """
        labels = self.ocr.read_chart_labels(image, text_regions, ocr_method=ocr_method)
        
        # Try to extract ylabel from left region if not found
        if not labels.get('ylabel') or labels.get('ylabel') == 'None' or labels.get('ylabel') == '':
            ylabel = self._extract_ylabel_from_left_region(image)
            if ylabel:
                labels['ylabel'] = ylabel
        
        return labels
    
    def _build_output(
        self, 
        labels: Dict[str, Any], 
        bars: List[Dict[str, Any]], 
        values: List[float], 
        categories: List[str]
    ) -> Dict[str, Any]:
        """
        Build final structured output.
        
        Args:
            labels: Extracted labels
            bars: Detected bars
            values: Calculated bar values
            categories: Extracted categories
            
        Returns:
            Structured data dict
        """
        structured_data = {
            'chart_type': 'bar_chart',
            'title': labels.get('title', 'Untitled'),
            'x_axis_label': labels.get('xlabel', 'X-axis'),
            'y_axis_label': labels.get('ylabel', 'Y-axis'),
            'data': []
        }
        
        for i, bar in enumerate(bars):
            cat = categories[i] if i < len(categories) else f'Category {i+1}'
            val = values[i] if i < len(values) else 0.0
            
            structured_data['data'].append({
                'category': cat,
                'value': val
            })
        
        return structured_data
    
    def extract(self, image_path: str, ocr_method: str = 'easyocr') -> Dict[str, Any]:
        """
        Main extraction pipeline.
        
        Args:
            image_path: Path to chart image file
            ocr_method: OCR engine to use ('easyocr' or 'tesseract')
        
        Returns:
            Dict with chart_type, title, x_axis_label, y_axis_label, data
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If ocr_method is invalid or image format unsupported
            InvalidImageError: If image cannot be loaded
        """
        # Validate ocr_method
        if ocr_method not in DEFAULT_LAYOUT.VALID_OCR_METHODS:
            raise ValueError(
                f"Invalid OCR method: {ocr_method}. "
                f"Valid options: {', '.join(DEFAULT_LAYOUT.VALID_OCR_METHODS)}"
            )
        
        # Step 1: Load and validate image
        image = self._load_and_validate_image(image_path)
        
        # Step 2: Detect components
        x_axis, y_axis, bars, text_regions = self._detect_components(image)
        
        # Step 3: Extract labels via OCR
        labels = self._extract_labels(image, text_regions, ocr_method)
        
        # Step 4: Calculate values
        values = self._calculate_bar_values(bars, y_axis, image, labels)
        
        # Step 5: Extract categories
        categories = self._extract_categories(labels, bars, image.shape[1], image.shape[0])
        
        # Step 6: Build and return output
        return self._build_output(labels, bars, values, categories)
    
    def _detect_y_axis_scale(
        self, 
        image: np.ndarray, 
        y_axis: Optional[np.ndarray], 
        labels: Dict[str, Any]
    ) -> Tuple[float, float, List[int], List[float]]:
        """
        Detect y-axis scale from ticks and labels.
        
        Args:
            image: Input image
            y_axis: Detected y-axis line or None
            labels: OCR labels dict
            
        Returns:
            Tuple of (y_min, y_max, tick_positions, tick_values)
        """
        h, w = image.shape[:2]
        
        # Tìm y-axis ticks (các số ở bên trái)
        y_ticks = []
        for val_info in labels.get('values', []):
            x, y = val_info['position']
            text = val_info['text']
            
            # Y-axis ticks ở bên trái
            if x < DEFAULT_LAYOUT.YTICK_REGION_MAX_X * w and val_info.get('is_number', False):
                try:
                    tick_value = float(text.replace(',', '').replace('%', ''))
                    y_ticks.append({
                        'value': tick_value,
                        'y_position': y
                    })
                except ValueError:
                    continue
        
        if len(y_ticks) < 2:
            # Fallback: giả sử scale 0-100
            return (
                DEFAULT_LAYOUT.FALLBACK_Y_MIN,
                DEFAULT_LAYOUT.FALLBACK_Y_MAX,
                [0, h - DEFAULT_LAYOUT.FALLBACK_BASELINE_OFFSET],
                [DEFAULT_LAYOUT.FALLBACK_Y_MIN, DEFAULT_LAYOUT.FALLBACK_Y_MAX]
            )
        
        # Sort theo y position (từ trên xuống - y nhỏ = trên cao)
        y_ticks.sort(key=lambda t: t['y_position'])
        
        # Lấy min và max
        y_min = min(t['value'] for t in y_ticks)
        y_max = max(t['value'] for t in y_ticks)
        
        # Nếu không có 0, thêm vào
        if y_min > 0:
            y_min = 0
        
        tick_positions = [t['y_position'] for t in y_ticks]
        tick_values = [t['value'] for t in y_ticks]
        
        return y_min, y_max, tick_positions, tick_values
    
    def _extract_bar_values_from_labels(
        self, 
        bars: List[Dict[str, Any]], 
        labels: Dict[str, Any], 
        image: np.ndarray
    ) -> List[Optional[float]]:
        """
        Extract values from labels above bars.
        
        Args:
            bars: List of detected bars
            labels: OCR labels dict
            image: Input image
            
        Returns:
            List of values (None for bars without labels)
        """
        h, w = image.shape[:2]
        values = []
        
        # Lấy các value labels (số ở trên bars)
        value_labels = [
            v for v in labels.get('values', [])
            if v.get('is_number', False)
        ]
        
        for bar in bars:
            bar_center_x = bar['center'][0]
            bar_top_y = bar['bbox'][1]  # Top of bar
            bar_bottom_y = bar['bbox'][1] + bar['bbox'][3]
            
            # Tìm value label gần bar nhất
            closest_label = None
            min_dist = float('inf')
            
            for label_info in value_labels:
                label_x, label_y = label_info['position']
                
                # Mở rộng vùng tìm kiếm
                x_dist = abs(label_x - bar_center_x)
                # Label có thể ở trên bar hoặc trong bar (value labels)
                y_dist = min(
                    abs(label_y - bar_top_y),  # Trên bar
                    abs(label_y - (bar_top_y + bar_bottom_y) / 2)  # Giữa bar
                )
                
                # Threshold linh hoạt hơn
                if x_dist < DEFAULT_LAYOUT.VALUE_X_THRESHOLD and y_dist < DEFAULT_LAYOUT.VALUE_Y_THRESHOLD:
                    total_dist = x_dist + y_dist * DEFAULT_LAYOUT.VALUE_Y_WEIGHT
                    if total_dist < min_dist:
                        min_dist = total_dist
                        closest_label = label_info
            
            if closest_label and min_dist < DEFAULT_LAYOUT.VALUE_TOTAL_DIST_THRESHOLD:
                # Parse value từ text
                try:
                    value_text = closest_label['text'].replace(',', '').replace('%', '').strip()
                    value = float(value_text)
                    values.append(round(value, 2))
                except ValueError:
                    # Fallback: None để tính từ pixel height
                    values.append(None)
            else:
                # Fallback: None để tính từ pixel height
                values.append(None)
        
        return values
    
    def _calculate_values_from_pixels(
        self, 
        bars: List[Dict[str, Any]], 
        y_axis: Optional[np.ndarray], 
        image: np.ndarray, 
        labels: Dict[str, Any]
    ) -> List[float]:
        """
        Calculate bar values from pixel heights using y-axis scale.
        
        Args:
            bars: List of detected bars
            y_axis: Detected y-axis line or None
            image: Input image
            labels: OCR labels dict
            
        Returns:
            List of calculated values
        """
        h, w = image.shape[:2]
        
        # Detect y-axis scale
        y_min, y_max, tick_positions, tick_values = self._detect_y_axis_scale(
            image, y_axis, labels
        )
        
        # Xác định baseline (x-axis y position)
        if y_axis is None:
            baseline_y = h - DEFAULT_LAYOUT.FALLBACK_BASELINE_OFFSET
        else:
            baseline_y = y_axis[1]
        
        # Tạo mapping function: pixel y -> value
        def pixel_to_value(pixel_y: int) -> float:
            """Convert pixel y position to actual value."""
            # Tìm 2 ticks gần nhất
            if len(tick_positions) < 2:
                # Linear interpolation với min/max
                value_range = y_max - y_min
                pixel_range = baseline_y - tick_positions[0] if tick_positions else h - DEFAULT_LAYOUT.FALLBACK_BASELINE_OFFSET
                ratio = (baseline_y - pixel_y) / pixel_range if pixel_range > 0 else 0
                return y_min + ratio * value_range
            
            # Interpolate giữa các ticks
            for i in range(len(tick_positions) - 1):
                y1 = tick_positions[i]
                y2 = tick_positions[i + 1]
                v1 = tick_values[i]
                v2 = tick_values[i + 1]
                
                if y1 <= pixel_y <= y2:
                    # Linear interpolation
                    ratio = (pixel_y - y1) / (y2 - y1) if (y2 - y1) > 0 else 0
                    return v1 + ratio * (v2 - v1)
            
            # Extrapolate nếu ngoài range
            if pixel_y < tick_positions[0]:
                # Above top tick
                y1, y2 = tick_positions[0], tick_positions[1] if len(tick_positions) > 1 else tick_positions[0]
                v1, v2 = tick_values[0], tick_values[1] if len(tick_values) > 1 else tick_values[0]
                ratio = (pixel_y - y1) / (y2 - y1) if (y2 - y1) > 0 else 0
                return v1 + ratio * (v2 - v1)
            else:
                # Below bottom tick (shouldn't happen for bars)
                return y_min
        
        values = []
        for bar in bars:
            x, y, w_bar, h_bar = bar['bbox']
            
            # Bar top position (highest point)
            bar_top_y = y
            
            # Calculate value
            value = pixel_to_value(bar_top_y)
            values.append(round(value, 2))
        
        return values
    
    def _calculate_bar_values(
        self, 
        bars: List[Dict[str, Any]], 
        y_axis: Optional[np.ndarray], 
        image: np.ndarray, 
        labels: Dict[str, Any]
    ) -> List[float]:
        """
        Calculate bar values: prioritize labels, fallback to pixel mapping.
        
        Args:
            bars: List of detected bars
            y_axis: Detected y-axis line or None
            image: Input image
            labels: OCR labels dict
            
        Returns:
            List of bar values
        """
        # Bước 1: Thử extract từ value labels (chính xác nhất)
        values_from_labels = self._extract_bar_values_from_labels(bars, labels, image)
        
        # Bước 2: Tính từ pixel height (fallback)
        values_from_pixels = self._calculate_values_from_pixels(bars, y_axis, image, labels)
        
        # Kết hợp: ưu tiên values từ labels, dùng pixel cho bars không có label
        final_values = []
        for i, bar in enumerate(bars):
            if i < len(values_from_labels) and values_from_labels[i] is not None:
                final_values.append(values_from_labels[i])
            elif i < len(values_from_pixels):
                final_values.append(values_from_pixels[i])
            else:
                final_values.append(0.0)
        
        return final_values
    
    def _extract_categories(
        self, 
        labels: Dict[str, Any], 
        bars: List[Dict[str, Any]], 
        img_width: int, 
        img_height: int
    ) -> List[str]:
        """
        Match category labels with bars using fuzzy matching.
        
        Args:
            labels: OCR labels dict
            bars: List of detected bars
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of category names
        """
        
        # Lấy tất cả texts ở vùng dưới (x-axis area)
        all_texts = labels.get('values', [])
        
        # Filter: chỉ lấy text ở dưới, không phải số, không phải x-axis label
        xlabel = labels.get('xlabel', '')
        category_candidates = []
        
        for text_info in all_texts:
            x, y = text_info['position']
            text = text_info['text']
            
            # Mở rộng vùng tìm kiếm (categories ở phần dưới)
            # Không phải số (categories thường là text)
            # Không phải x-axis label
            if (y > DEFAULT_LAYOUT.CATEGORY_REGION_MIN_Y * img_height and
                not text_info.get('is_number', False) and
                text != xlabel and
                len(text) > 0):
                category_candidates.append(text_info)
        
        # Sort theo x position (trái sang phải)
        category_candidates = sorted(category_candidates, key=lambda t: t['position'][0])
        
        # ========== CẢI THIỆN: Fuzzy matching với common month abbreviations và OCR errors ==========
        # Common patterns để sửa lỗi OCR
        month_patterns = {
            # Month abbreviations
            'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
            'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
            'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
            # Common OCR errors for months
            'jui': 'Jul', 'ju1': 'Jul', 'juI': 'Jul', 'jul1': 'Jul',
            # Full month names
            'january': 'Jan', 'february': 'Feb', 'march': 'Mar',
            'april': 'Apr', 'june': 'Jun', 'july': 'Jul',
            'august': 'Aug', 'september': 'Sep', 'october': 'Oct',
            'november': 'Nov', 'december': 'Dec',
            # Common OCR errors for categories (rotated text)
            'catl': 'Cat1', 'cat1': 'Cat1', 'cat3': 'Cat3', '[cat4': 'Cat4',
            'cat4': 'Cat4', 'cat5': 'Cat5', 'cato': 'Cat0', 'cat0': 'Cat0',
            'cat8l': 'Cat8', 'cat8': 'Cat8', 'cat7': 'Cat7',
            # Region names OCR errors
            'eastl': 'East', 'east': 'East', 'westl': 'West', 'west': 'West',
            'centrall': 'Central', 'central': 'Central',
            'north': 'North', 'south': 'South',
            # Product names
            'producte': 'Products', 'product': 'Product', 'products': 'Products',
            # Student names
            'student': 'Student', 'students': 'Students'
        }
        
        def normalize_text(text: str) -> str:
            """Normalize text and fix common OCR errors."""
            text_lower = text.lower().strip()
            # Thử match với month patterns
            for pattern, correct in month_patterns.items():
                if pattern in text_lower or text_lower in pattern:
                    return correct
            return text
        
        # Match với bars
        categories = []
        used_candidates = set()
        
        for i, bar in enumerate(bars):
            bar_center_x = bar['center'][0]
            
            # Tìm text gần nhất
            closest_text = None
            min_dist = float('inf')
            closest_idx = -1
            
            for idx, text_info in enumerate(category_candidates):
                if idx in used_candidates:
                    continue
                
                text_x = text_info['position'][0]
                dist = abs(text_x - bar_center_x)
                
                # Dynamic threshold
                threshold = max(
                    DEFAULT_LAYOUT.MIN_SPACING_THRESHOLD,
                    img_width / (len(bars) * DEFAULT_LAYOUT.SPACING_FACTOR)
                ) if len(bars) > 0 else DEFAULT_LAYOUT.MIN_SPACING_THRESHOLD
                
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    closest_text = text_info
                    closest_idx = idx
            
            if closest_text and min_dist < threshold:
                # Normalize text để sửa OCR errors
                normalized = normalize_text(closest_text['text'])
                categories.append(normalized)
                used_candidates.add(closest_idx)
            else:
                # Fallback: dùng index
                categories.append(f'Category {i+1}')
        
        return categories