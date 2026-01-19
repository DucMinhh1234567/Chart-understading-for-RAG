"""Simplified bar chart extractor used in module 5 tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.extraction.ocr_engine import OCREngine
from src.preprocessing.chart_detector import ChartComponentDetector
from src.preprocessing.image_utils import ImagePreprocessor

if TYPE_CHECKING:
    from pathlib import Path


class BarChartExtractor:
    """Extract structured data from bar chart images.

    This version keeps a simpler implementation but preserves the public
    interface expected by the module_5 notebook (including the
    ``ocr_method`` argument).
    """

    def __init__(
        self,
        detector: ChartComponentDetector | None = None,
        ocr_engine: OCREngine | None = None,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.detector = detector or ChartComponentDetector(self.preprocessor)
        self.ocr = ocr_engine or OCREngine(languages=["en", "vi"], gpu=False)

    def extract(self, image_path: str | Path, ocr_method: str = "easyocr") -> dict:
        """Main extraction pipeline. Returns structured data.

        Parameters
        ----------
        image_path:
            Path to the bar chart image file.
        ocr_method:
            OCR method to use (kept for compatibility). The current
            implementation forwards this to the OCR engine.
        """
        # Load image
        image = self.preprocessor.load_image(str(image_path))

        # Step 1: Detect components
        x_axis, y_axis = self.detector.detect_axes(image)
        bars = self.detector.detect_bars(image, x_axis, y_axis)
        text_regions = self.detector.detect_text_regions(image)

        # Step 2: OCR
        labels = self.ocr.read_chart_labels(image, text_regions, ocr_method=ocr_method)

        # Step 3: Calculate values (sử dụng y_ticks nếu có để tính scale)
        values = self._calculate_bar_values(bars, y_axis, image.shape[0], labels.get("y_ticks", []))

        # Step 4: Match categories with bars
        categories = self._extract_categories(labels, bars, image.shape[1], image.shape[0])

        # Step 5: Create structured data
        structured_data: dict = {
            "chart_type": "bar_chart",
            "title": labels.get("title", "Untitled"),
            "x_axis_label": labels.get("xlabel", "X-axis"),
            "y_axis_label": labels.get("ylabel", "Y-axis"),
            "data": [],
        }

        for i, bar in enumerate(bars):
            cat = categories[i] if i < len(categories) else f"Category {i+1}"
            val = values[i] if i < len(values) else 0.0

            structured_data["data"].append(
                {
                    "category": cat,
                    "value": float(val),
                }
            )

        return structured_data

    def _calculate_bar_values(self, bars, y_axis, img_height, y_ticks=None):
        """Tính giá trị của mỗi bar dựa trên chiều cao pixel.
        
        Nếu có y_ticks, sẽ convert sang giá trị thật theo scale.
        Nếu không, normalize về 0-100.
        """
        if not bars:
            return []

        if y_ticks is None:
            y_ticks = []

        # Xác định baseline (X-axis Y coordinate)
        if not y_axis:
            # Fallback: dùng bottom của image làm baseline
            baseline_y = img_height - 50
        else:
            # Y-axis là (x1, y1, x2, y2), baseline là Y coordinate của X-axis
            # Thường là max(y1, y2) hoặc có thể dùng max bottom của bars
            baseline_y = max((bar["bbox"][1] + bar["bbox"][3]) for bar in bars) if bars else img_height - 50

        # Nếu có y_ticks, tính scale để convert sang giá trị thật
        scale_info = None
        if len(y_ticks) >= 2:
            # Sort theo Y position (từ trên xuống dưới)
            sorted_ticks = sorted(y_ticks, key=lambda t: t["position"][1])
            
            # Lấy tick trên cùng và dưới cùng
            top_tick = sorted_ticks[0]
            bottom_tick = sorted_ticks[-1]
            
            top_y = top_tick["position"][1]
            bottom_y = bottom_tick["position"][1]
            pixel_range = bottom_y - top_y
            
            top_value = top_tick["value"]
            bottom_value = bottom_tick["value"]
            value_range = bottom_value - top_value
            
            if pixel_range > 0 and value_range != 0:
                scale_info = {
                    "min": top_value,
                    "max": bottom_value,
                    "pixel_min": top_y,
                    "pixel_max": bottom_y,
                    "pixel_range": pixel_range,
                    "value_range": value_range,
                }

        values: list[float] = []

        for bar in bars:
            x, y, w, h = bar["bbox"]

            # Bar height in pixels (from baseline to bar top)
            bar_top = y
            bar_height_px = max(0.0, float(baseline_y - bar_top))

            # Nếu có scale_info, convert sang giá trị thật
            if scale_info:
                # Tính vị trí Y của bar top trong hệ tọa độ của scale
                # Scale từ top_y (min) đến bottom_y (max)
                if bar_top < scale_info["pixel_min"]:
                    # Bar cao hơn top tick
                    value = scale_info["max"]
                elif bar_top > scale_info["pixel_max"]:
                    # Bar thấp hơn bottom tick
                    value = scale_info["min"]
                else:
                    # Linear interpolation
                    normalized = (scale_info["pixel_max"] - bar_top) / scale_info["pixel_range"]
                    value = scale_info["min"] + normalized * scale_info["value_range"]
                values.append(round(value, 2))
            else:
                # Không có scale, dùng pixel height
                values.append(bar_height_px)

        # Nếu không có scale_info, normalize về 0-100
        if not scale_info:
            max_height_px = max(values) if values else 1.0
            if max_height_px <= 0:
                return [0.0 for _ in values]
            values = [round((v / max_height_px) * 100.0, 2) for v in values]

        return values

    def _extract_categories(self, labels, bars, img_width, img_height):
        """Match category labels với bars dựa trên vị trí X."""
        # Lấy các text ở dưới (x-axis labels)
        value_texts = labels.get("values", [])

        # Filter: chỉ lấy text ở phía dưới (dùng img_height)
        bottom_texts = [v for v in value_texts if v["position"][1] > img_height * 0.7]

        # Filter cải thiện: loại bỏ các label không phải category
        def is_valid_category(text: str) -> bool:
            """Kiểm tra xem text có phải là category label hợp lệ không."""
            text = text.strip()
            if not text or len(text) > 25:  # Quá dài
                return False
            
            # Loại bỏ text có quá nhiều số (thường là tick values hoặc giá trị)
            words = text.split()
            num_numbers = sum(1 for word in words if any(c.isdigit() for c in word))
            if num_numbers > len(words) * 0.5:  # Nếu > 50% là số
                return False
            
            # Loại bỏ text có quá nhiều từ (thường là title hoặc label dài)
            if len(words) > 4:  # Category thường chỉ 1-3 từ
                return False
            
            # Loại bỏ text chỉ toàn số (là tick value, không phải category)
            if text.replace(".", "").replace(",", "").replace("-", "").isdigit():
                return False
            
            return True

        cleaned_texts = [v for v in bottom_texts if is_valid_category(v["text"])]
        if cleaned_texts:
            bottom_texts = cleaned_texts

        # Sort theo position x
        bottom_texts = sorted(bottom_texts, key=lambda t: t["position"][0])

        categories: list[str] = []
        # Tính threshold động dựa trên khoảng cách giữa các bars
        if len(bars) > 1:
            bar_centers = [bar["center"][0] for bar in bars]
            bar_centers.sort()
            avg_bar_spacing = sum(bar_centers[i+1] - bar_centers[i] for i in range(len(bar_centers)-1)) / (len(bar_centers) - 1)
            max_distance = max(50.0, avg_bar_spacing * 0.4)  # 40% khoảng cách giữa các bars
        else:
            max_distance = max(50.0, img_width * 0.1)  # Fallback: 10% chiều rộng ảnh

        # Track các label đã được sử dụng để tránh duplicate
        used_labels = set()

        for i, bar in enumerate(bars):
            bar_center_x = bar["center"][0]

            # Tìm text gần bar center nhất và chưa được sử dụng
            closest_text: str | None = None
            closest_info = None
            min_dist = float("inf")

            for text_info in bottom_texts:
                text_x = text_info["position"][0]
                dist = abs(text_x - bar_center_x)

                # Chỉ xét label chưa được sử dụng
                text_key = text_info["text"].strip()
                if text_key not in used_labels and dist < min_dist:
                    min_dist = dist
                    closest_text = text_key
                    closest_info = text_info

            if closest_text and min_dist < max_distance:
                categories.append(closest_text)
                used_labels.add(closest_text)
            else:
                categories.append(f"Category {i+1}")

        return categories