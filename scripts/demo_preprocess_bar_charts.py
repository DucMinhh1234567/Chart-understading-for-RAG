"""Demo script to visualize input/output of `ImagePreprocessor`.

This script:
- Lấy 10 ảnh đầu tiên trong `data/raw/bar_charts`
- Chạy pipeline tiền xử lý: load -> remove_noise -> enhance_contrast
- Lưu kết quả vào thư mục `temp` (ở root project `bleh/temp`):
  * `preprocessed_*.png`: Ảnh RGB đã qua tiền xử lý
  * `binary_*.png`: Ảnh nhị phân (binarize) - giúp thấy rõ foreground/background
  * `edges_*.png`: Ảnh biên (Canny edge detection) - giúp thấy rõ các đường viền

So sánh các file này để hiểu rõ hơn về hiệu quả của tiền xử lý.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


# Đảm bảo có thể import được package `src` khi chạy file trực tiếp từ thư mục `scripts`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.image_utils import ImagePreprocessor


def ensure_temp_dir(base_dir: Path) -> Path:
    """Đảm bảo tồn tại thư mục `temp` dưới root project."""
    temp_dir = base_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_first_n_images(images_dir: Path, n: int = 10) -> list[Path]:
    """Lấy N ảnh đầu tiên (sắp xếp tên file) trong thư mục."""
    all_pngs = sorted(images_dir.glob("chart_*.png"))
    return all_pngs[:n]


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    """Lưu ảnh RGB (numpy array) ra file PNG bằng OpenCV.

    OpenCV kỳ vọng ảnh dạng BGR, nên cần convert RGB -> BGR trước khi ghi.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def main() -> None:
    # Xác định root project: file này nằm ở `bleh/scripts`, nên root là 1 level phía trên.
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    # Thư mục input và output
    bar_charts_dir = project_root / "data" / "raw" / "bar_charts"
    temp_dir = ensure_temp_dir(project_root)

    print(f"Input folder : {bar_charts_dir}")
    print(f"Output folder: {temp_dir}")

    image_paths = get_first_n_images(bar_charts_dir, n=10)
    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục bar_charts.")
        return

    preprocessor = ImagePreprocessor()

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx:02d}] Processing {img_path.name}...")

        # Option 1: Sử dụng pipeline cải tiến với blur nhẹ hơn (3x3 thay vì 5x5)
        try:
            enhanced = preprocessor.preprocess_pipeline(
                str(img_path),
                blur_ksize=(3, 3),  # Giảm blur để giữ chi tiết
                apply_blur=True,
                normalize=False,  # Không resize để giữ nguyên kích thước gốc
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Lỗi khi xử lý {img_path.name}: {exc}")
            continue

        # 2) Lưu ảnh đã enhance contrast vào temp
        out_name = f"preprocessed_{img_path.name}"
        out_path = temp_dir / out_name
        save_rgb_image(out_path, enhanced)
        print(f"  -> Saved preprocessed image: {out_path.name}")

        # 3) Tạo và lưu ảnh nhị phân (binary) để thấy rõ hiệu quả tiền xử lý
        try:
            binary = preprocessor.binarize(enhanced, method="otsu")
            binary_path = temp_dir / f"binary_{img_path.name}"
            cv2.imwrite(str(binary_path), binary)
            print(f"  -> Saved binary image: {binary_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> Lỗi khi tạo binary image: {exc}")

        # 4) Tạo và lưu ảnh biên với adaptive Canny (tốt hơn fixed threshold)
        try:
            edges_adaptive = preprocessor.detect_edges_adaptive(enhanced)
            edges_path = temp_dir / f"edges_{img_path.name}"
            cv2.imwrite(str(edges_path), edges_adaptive)
            print(f"  -> Saved edges image (adaptive): {edges_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> Lỗi khi tạo edges image: {exc}")

    print("\n" + "=" * 60)
    print("Hoàn thành demo tiền xử lý 10 ảnh bar chart đầu tiên.")
    print(f"\nĐã tạo các file trong thư mục: {temp_dir}")
    print("  - preprocessed_*.png: Ảnh RGB đã qua tiền xử lý (blur 3x3 + CLAHE)")
    print("  - binary_*.png: Ảnh nhị phân (Otsu thresholding)")
    print("  - edges_*.png: Ảnh biên (Adaptive Canny edge detection)")
    print("\nCải tiến so với trước:")
    print("  ✓ Blur kernel giảm từ 5x5 → 3x3 (giữ chi tiết tốt hơn)")
    print("  ✓ Canny edge detection dùng adaptive threshold (tự động điều chỉnh)")
    print("\nMở các file này để so sánh và thấy rõ hiệu quả tiền xử lý!")
    print("=" * 60)


if __name__ == "__main__":
    main()

