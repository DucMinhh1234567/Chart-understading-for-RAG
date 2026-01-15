"""OCR engine wrapper for chart-understanding.

This module provides :class:`OCREngine` for reading text from chart images
using EasyOCR and Tesseract OCR engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import easyocr
except ImportError:
    easyocr = None  # type: ignore[assignment, unused-ignore]

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment, unused-ignore]


class OCREngine:
    """OCR engine wrapper for reading text from chart images.

    Supports both EasyOCR and Tesseract OCR engines for text extraction
    from chart components (titles, labels, values).

    Attributes
    ----------
    easyocr_reader: easyocr.Reader | None
        EasyOCR reader instance if EasyOCR is available.
    languages: list[str]
        List of language codes for OCR.
    """

    def __init__(self, languages: Sequence[str] | None = None, gpu: bool = False) -> None:
        """Initialize the OCR engine.

        Parameters
        ----------
        languages:
            List of language codes (e.g., ['en', 'vi']). Defaults to ['en', 'vi'].
        gpu:
            Whether to use GPU for EasyOCR. Default False for compatibility.

        Raises
        ------
        ImportError
            If neither EasyOCR nor pytesseract is installed.
        """
        self.languages = list(languages) if languages is not None else ["en", "vi"]
        self.gpu = gpu

        # Initialize EasyOCR if available
        self.easyocr_reader: easyocr.Reader | None = None
        if easyocr is not None:
            try:
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=gpu)
            except Exception as exc:
                # If EasyOCR fails to initialize, continue without it
                print(f"Warning: Failed to initialize EasyOCR: {exc}")
                self.easyocr_reader = None

        # Check if at least one OCR engine is available
        if self.easyocr_reader is None and pytesseract is None:
            raise ImportError(
                "At least one OCR engine (EasyOCR or pytesseract) must be installed. "
                "Install with: pip install easyocr or pip install pytesseract"
            )

    def read_text_easyocr(
        self, image_region: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[dict]:
        """Read text from image region using EasyOCR.

        Parameters
        ----------
        image_region:
            Image region as numpy array (RGB or grayscale).
        confidence_threshold:
            Minimum confidence score to include a text detection (0.0-1.0).

        Returns
        -------
        list[dict]
            List of text detections, each containing:
            - 'text': Detected text string
            - 'confidence': Confidence score (0.0-1.0)
            - 'bbox': Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Raises
        ------
        RuntimeError
            If EasyOCR is not available or fails to read text.
        """
        if self.easyocr_reader is None:
            raise RuntimeError("EasyOCR is not available. Install with: pip install easyocr")

        try:
            # Ensure image is in correct format (numpy array)
            if not isinstance(image_region, np.ndarray):
                raise ValueError("image_region must be a numpy array")

            # EasyOCR expects RGB or grayscale images
            if image_region.ndim == 3 and image_region.shape[2] == 3:
                # Already RGB
                img = image_region
            elif image_region.ndim == 2:
                # Grayscale - convert to RGB
                img = np.stack([image_region] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported image shape: {image_region.shape}")

            # Read text using EasyOCR
            results = self.easyocr_reader.readtext(img)

            # Process and filter results
            detections: list[dict] = []
            for detection in results:
                bbox, text, confidence = detection

                # Filter by confidence
                if confidence < confidence_threshold:
                    continue

                # Clean and normalize text
                text_clean = text.strip()

                # Skip empty text
                if not text_clean:
                    continue

                detections.append(
                    {
                        "text": text_clean,
                        "confidence": float(confidence),
                        "bbox": bbox,  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    }
                )

            return detections

        except Exception as exc:
            raise RuntimeError(f"Failed to read text with EasyOCR: {exc}") from exc

    def read_text_tesseract(
        self, image_region: np.ndarray, lang: str | None = None
    ) -> list[dict]:
        """Read text from image region using Tesseract OCR.

        Parameters
        ----------
        image_region:
            Image region as numpy array (RGB or grayscale).
        lang:
            Language code (e.g., 'vie+eng'). If None, constructs from self.languages.

        Returns
        -------
        list[dict]
            List of text detections, each containing:
            - 'text': Detected text string
            - 'confidence': Confidence score (0.0-100.0, normalized to 0.0-1.0)
            - 'bbox': Bounding box coordinates [x, y, w, h]

        Raises
        ------
        RuntimeError
            If pytesseract is not available or fails to read text.
        """
        if pytesseract is None:
            raise RuntimeError(
                "pytesseract is not available. Install with: pip install pytesseract"
            )

        try:
            # Ensure image is in correct format
            if not isinstance(image_region, np.ndarray):
                raise ValueError("image_region must be a numpy array")

            # Convert numpy array to PIL Image
            if image_region.ndim == 3 and image_region.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image_region.astype(np.uint8))
            elif image_region.ndim == 2:
                # Grayscale image
                pil_image = Image.fromarray(image_region.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported image shape: {image_region.shape}")

            # Construct language code if not provided
            if lang is None:
                # Map language codes (e.g., 'vi' -> 'vie', 'en' -> 'eng')
                lang_map = {"en": "eng", "vi": "vie", "vie": "vie", "eng": "eng"}
                lang_codes = [lang_map.get(l.lower(), l) for l in self.languages]
                lang = "+".join(lang_codes)

            # Get detailed data including bounding boxes
            try:
                data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "Tesseract OCR not found. Please install Tesseract OCR on your system."
                )

            detections: list[dict] = []

            # Process Tesseract output
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                # Skip empty text or low confidence
                if not text or conf < 0:
                    continue

                # Normalize confidence to 0.0-1.0 range
                confidence = float(conf) / 100.0

                # Get bounding box
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                # Convert to format similar to EasyOCR: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                detections.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

            return detections

        except Exception as exc:
            raise RuntimeError(f"Failed to read text with Tesseract: {exc}") from exc

    def read_chart_labels(
        self,
        image: np.ndarray,
        text_regions: Sequence[tuple[int, int, int, int]],
        ocr_method: str = "easyocr",
    ) -> dict:
        """Read and classify chart labels based on text region positions.

        Parameters
        ----------
        image:
            Full chart image (RGB or grayscale).
        text_regions:
            List of text region bounding boxes (x, y, w, h) from text detection.
        ocr_method:
            OCR method to use: 'easyocr' or 'tesseract'. Defaults to 'easyocr'.

        Returns
        -------
        dict
            Dictionary containing classified labels:
            - 'title': Detected title text (str)
            - 'xlabel': Detected X-axis label text (str)
            - 'ylabel': Detected Y-axis label text (str)
            - 'values': List of value detections [{'text': str, 'position': (x,y)}, ...]
        """
        try:
            # Get image dimensions
            if image.ndim == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape

            # Initialize result dictionary
            result: dict = {
                "title": "",
                "xlabel": "",
                "ylabel": "",
                "values": [],
            }

            # Select OCR method
            read_method = self.read_text_easyocr if ocr_method.lower() == "easyocr" else self.read_text_tesseract

            # Process each text region
            for bbox in text_regions:
                x, y, region_w, region_h = bbox

                # Extract region of interest
                if image.ndim == 3:
                    roi = image[y : y + region_h, x : x + region_w]
                else:
                    roi = image[y : y + region_h, x : x + region_w]

                # Skip empty regions
                if roi.size == 0:
                    continue

                # Calculate region center position (normalized 0-1)
                center_x = (x + region_w / 2) / w
                center_y = (y + region_h / 2) / h

                try:
                    # Read text from region
                    detections = read_method(roi)

                    if len(detections) == 0:
                        continue

                    # Combine all text in region
                    texts = [d["text"] for d in detections]
                    combined_text = " ".join(texts).strip()

                    if not combined_text:
                        continue

                    # Classify region based on position
                    # Title: near top (y < 0.2 * height)
                    if center_y < 0.2:
                        if not result["title"]:
                            result["title"] = combined_text
                        else:
                            # If multiple title candidates, keep the first one
                            pass

                    # X-axis label: bottom region, not too far left (y > 0.85 * height, x > 0.3 * width)
                    elif center_y > 0.85 and center_x > 0.3:
                        if not result["xlabel"]:
                            result["xlabel"] = combined_text

                    # Y-axis label: left region, not too far down (x < 0.15 * width, y < 0.7 * height)
                    elif center_x < 0.15 and center_y < 0.7:
                        if not result["ylabel"]:
                            result["ylabel"] = combined_text

                    # Values: other regions (typically bar values, tick labels, etc.)
                    else:
                        result["values"].append(
                            {
                                "text": combined_text,
                                "position": (int(x + region_w / 2), int(y + region_h / 2)),
                            }
                        )

                except Exception as exc:
                    # Skip region if OCR fails
                    print(f"Warning: Failed to read text from region {bbox}: {exc}")
                    continue

            return result

        except Exception as exc:
            raise RuntimeError(f"Failed to read chart labels: {exc}") from exc