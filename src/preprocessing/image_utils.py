"""Image preprocessing utilities for chart-understanding.

This module provides a convenience class :class:`ImagePreprocessor`
for common preprocessing steps used before chart detection / extraction.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


MethodType = Literal["otsu", "adaptive"]


class ImagePreprocessor:
    """Utility class for image preprocessing operations.

    All methods are implemented as instance methods to allow future
    configuration (e.g. configurable blur size, thresholds, etc.).
    """

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk and convert it to RGB.

        Parameters
        ----------
        image_path:
            Path to the input image file.

        Returns
        -------
        np.ndarray
            Image array in RGB format with shape (H, W, 3).

        Raises
        ------
        FileNotFoundError
            If the image cannot be loaded (e.g. file does not exist or is unreadable).
        """
        try:
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return image_rgb
        except FileNotFoundError:
            # Re-raise to keep the error explicit for callers.
            raise
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(f"Failed to read image '{image_path}': {exc}") from exc

    def remove_noise(self, image: np.ndarray, ksize: tuple[int, int] = (3, 3)) -> np.ndarray:
        """Apply Gaussian blur to reduce noise.

        Parameters
        ----------
        image:
            Input image (RGB or grayscale) as a NumPy array.
        ksize:
            Kernel size for Gaussian blur. Default (3, 3) preserves more detail
            than (5, 5). Use larger kernel for very noisy images.

        Returns
        -------
        np.ndarray
            Denoised image with the same shape as input.
        """
        try:
            return cv2.GaussianBlur(image, ksize=ksize, sigmaX=0)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to remove noise: {exc}") from exc

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE in LAB color space.

        Steps
        -----
        1. Convert RGB -> LAB
        2. Apply CLAHE on L channel
        3. Merge channels and convert back LAB -> RGB

        Parameters
        ----------
        image:
            Input image in RGB format (H, W, 3).

        Returns
        -------
        np.ndarray
            Contrast-enhanced RGB image.
        """
        try:
            # Ensure 3-channel image
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("enhance_contrast expects an RGB image with shape (H, W, 3).")

            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            merged = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            return enhanced_rgb
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to enhance contrast: {exc}") from exc

    def binarize(self, image: np.ndarray, method: MethodType = "otsu") -> np.ndarray:
        """Binarize image using Otsu or adaptive thresholding.

        Parameters
        ----------
        image:
            Input image (RGB or grayscale).
        method:
            Thresholding method to use. Supported values:

            - ``\"otsu\"``: Otsu's global thresholding.
            - ``\"adaptive\"``: Adaptive mean thresholding.

        Returns
        -------
        np.ndarray
            Binary image (grayscale, single channel) with values 0 or 255.

        Raises
        ------
        ValueError
            If an unsupported method is provided.
        """
        try:
            # Convert to grayscale if needed
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            if method == "otsu":
                _, binary = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
            elif method == "adaptive":
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
            else:
                raise ValueError(f"Unsupported binarization method: {method}")

            return binary
        except ValueError:
            # Re-throw explicit configuration errors.
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to binarize image: {exc}") from exc

    def detect_edges(self, image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
        """Detect edges using Canny edge detection.

        Parameters
        ----------
        image:
            Input image (RGB or grayscale).
        threshold1:
            First threshold for the hysteresis procedure.
        threshold2:
            Second threshold for the hysteresis procedure.

        Returns
        -------
        np.ndarray
            Edge map as a single channel binary image.
        """
        try:
            # Canny expects grayscale
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)
            return edges
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to detect edges: {exc}") from exc

    def detect_edges_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny with adaptive thresholds based on image statistics.

        This method automatically calculates optimal thresholds based on the median
        pixel value, making it more robust across different image conditions.

        Parameters
        ----------
        image:
            Input image (RGB or grayscale).

        Returns
        -------
        np.ndarray
            Edge map as a single channel binary image.
        """
        try:
            # Canny expects grayscale
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Calculate adaptive thresholds based on median
            median = np.median(gray)
            lower = int(max(0, 0.7 * median))
            upper = int(min(255, 1.3 * median))

            edges = cv2.Canny(gray, lower, upper)
            return edges
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to detect edges adaptively: {exc}") from exc

    def normalize_size(self, image: np.ndarray, max_size: int = 2048) -> np.ndarray:
        """Resize image to a maximum dimension while preserving aspect ratio.

        Useful for standardizing input sizes for models while maintaining
        image quality. Uses INTER_AREA interpolation for downscaling.

        Parameters
        ----------
        image:
            Input image (RGB or grayscale).
        max_size:
            Maximum dimension (width or height). If image is smaller, no resize.

        Returns
        -------
        np.ndarray
            Resized image with same number of channels, or original if already smaller.
        """
        try:
            h, w = image.shape[:2]
            if max(h, w) <= max_size:
                return image

            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to normalize size: {exc}") from exc

    def preprocess_pipeline(
        self,
        image_path: str,
        blur_ksize: tuple[int, int] = (3, 3),
        apply_blur: bool = True,
        normalize: bool = False,
        max_size: int = 2048,
    ) -> np.ndarray:
        """Full preprocessing pipeline: load → (optional resize) → denoise → enhance_contrast.

        Parameters
        ----------
        image_path:
            Path to the input image file.
        blur_ksize:
            Kernel size for Gaussian blur. Default (3, 3) preserves more detail.
        apply_blur:
            Whether to apply noise reduction. Set False for already clean images.
        normalize:
            Whether to resize image to max_size. Useful for standardizing inputs.
        max_size:
            Maximum dimension when normalize=True.

        Returns
        -------
        np.ndarray
            Preprocessed RGB image.
        """
        # Mỗi bước đều có error handling riêng; để lỗi propagate lên caller
        image = self.load_image(image_path)

        if normalize:
            image = self.normalize_size(image, max_size=max_size)

        if apply_blur:
            denoised = self.remove_noise(image, ksize=blur_ksize)
        else:
            denoised = image

        enhanced = self.enhance_contrast(denoised)
        return enhanced

    def preprocess_for_chart_detection(
        self,
        image_path: str,
        blur_ksize: tuple[int, int] = (3, 3),
        normalize: bool = True,
        max_size: int = 2048,
        binarize_method: MethodType = "otsu",
    ) -> dict[str, np.ndarray]:
        """Optimized preprocessing pipeline for chart detection/extraction tasks.

        Returns multiple image versions optimized for different downstream tasks:
        - RGB: For OCR, color detection, general visualization
        - Binary: For component detection (bars, axes, regions)
        - Edges: For line/axis detection using adaptive Canny

        Parameters
        ----------
        image_path:
            Path to the input image file.
        blur_ksize:
            Kernel size for Gaussian blur. Default (3, 3) preserves detail.
        normalize:
            Whether to resize image to max_size.
        max_size:
            Maximum dimension when normalize=True.
        binarize_method:
            Method for binarization ("otsu" or "adaptive").

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - "rgb": Preprocessed RGB image
            - "binary": Binary image for component detection
            - "edges": Edge map for line detection
        """
        # Load and preprocess RGB image
        rgb = self.preprocess_pipeline(
            image_path,
            blur_ksize=blur_ksize,
            apply_blur=True,
            normalize=normalize,
            max_size=max_size,
        )

        # Generate binary and edge versions
        binary = self.binarize(rgb, method=binarize_method)
        edges = self.detect_edges_adaptive(rgb)

        return {
            "rgb": rgb,
            "binary": binary,
            "edges": edges,
        }
