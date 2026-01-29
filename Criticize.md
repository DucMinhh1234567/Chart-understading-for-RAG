# Đánh giá Repo Chart Understanding

## 📊 Tổng quan

Đây là một dự án CV (Computer Vision) + OCR để hiểu và trích xuất dữ liệu từ biểu đồ (bar charts). Có tiềm năng tốt nhưng cần nhiều cải thiện về cấu trúc và kỹ thuật.

---

## ✅ Điểm mạnh

### 1. **Cấu trúc project rõ ràng**
- Phân chia module hợp lý: `preprocessing`, `extraction`, `models`, `text_generation`
- Có `scripts/` riêng cho demo và utilities
- Tách biệt code và data (`data/raw/`, `data/annotations/`)

### 2. **Pipeline xử lý có logic**
```
Image → Preprocess → Detect Components → OCR → Extract Data
```
- Preprocessing: noise removal, contrast enhancement, edge detection
- Detection: axes, bars, text regions
- OCR: EasyOCR + Tesseract fallback
- Validation pipeline với Chain of Responsibility pattern

### 3. **Có validation pipeline**
- `BarValidator` với nhiều validators độc lập:
  - Width, Area, AspectRatio, Spacing validators
  - Có thể compose và test riêng từng validator
- Giúp filter false positives tốt hơn

### 4. **Adaptive configuration**
- `ChartDetectorConfig` tự điều chỉnh thresholds theo kích thước ảnh
- Hợp lý cho việc xử lý charts với độ phân giải khác nhau

### 5. **Error handling**
- Custom exceptions: `InvalidImageError`, `AxisDetectionError`, `BarDetectionError`
- Logging đầy đủ cho debugging

---

## ❌ Điểm yếu nghiêm trọng

### 1. **Code chất lượng thấp - Vi phạm nguyên tắc cơ bản**

#### **a) Hàm quá dài, làm quá nhiều việc**
```python
# src/extraction/bar_extractor.py: 450+ lines
def extract(self, image_path, ocr_method='easyocr'):
    # Load image
    # Detect components
    # OCR
    # Calculate values
    # Extract categories
    # Create structured data
    # ... 100+ lines
```
**Vấn đề:** Vi phạm Single Responsibility Principle (SRP). Một hàm làm 6-7 việc khác nhau.

**Giải pháp:**
```python
def extract(self, image_path, ocr_method='easyocr'):
    image = self._load_image(image_path)
    components = self._detect_components(image)
    labels = self._extract_labels(image, components, ocr_method)
    data = self._build_structured_data(components, labels)
    return data
```

#### **b) Magic numbers và hardcoded values tràn lan**
```python
# Ở khắp nơi:
if y < 0.2 * h and val_info.get('is_number', False):
if x_pos < 0.3 * w and y_pos < 0.85 * h:
if dist < threshold and dist < min_dist:
threshold = max(40, img_width / (len(bars) * 2.5))
```

**Vấn đề:** Không thể test, không thể tune, không hiểu ý nghĩa.

**Giải pháp:**
```python
class LayoutConfig:
    TITLE_REGION_TOP = 0.2
    YLABEL_REGION_LEFT = 0.3
    XLABEL_REGION_BOTTOM = 0.85
    MIN_SPACING_THRESHOLD = 40
    SPACING_FACTOR = 2.5
```

#### **c) Duplicate code**
```python
# Trong OCREngine.read_chart_labels():
# Group texts theo dòng (40 lines)
title_lines = {}
for text_info in title_region_texts:
    y_pos = text_info['position'][1]
    y_line = round(y_pos / 20) * 20
    if y_line not in title_lines:
        title_lines[y_line] = []
    title_lines[y_line].append(text_info)

# Sau đó lại có logic tương tự cho ylabel...
ylabel_candidates = []
for t in all_texts:
    x_pos, y_pos = t['position']
    if (x_pos < 0.3 * w and ...):
        # Lặp lại logic grouping
```

**Giải pháp:** Extract thành helper method:
```python
def _group_texts_by_line(self, texts, axis='y', tolerance=20):
    """Group texts into lines based on position"""
    lines = {}
    for text in texts:
        pos = text['position'][1 if axis == 'y' else 0]
        line_key = round(pos / tolerance) * tolerance
        lines.setdefault(line_key, []).append(text)
    return lines
```

### 2. **Thiếu tests hoàn toàn**
- Không có test nào trong repo
- Code phức tạp như `SpacingValidator`, `_detect_y_axis_scale()` PHẢI có tests
- Không thể refactor an toàn

**Tác hại:**
- Bug ẩn khó phát hiện
- Sợ refactor → code càng ngày càng tệ
- Không thể validate improvements

### 3. **Dependency hell**
```python
# requirements.txt
opencv-python
numpy
scipy
pandas
matplotlib
pillow
pytesseract
easyocr
scikit-image
scikit-learn
torch
torchvision
transformers
timm
ultralytics  # ??? Không thấy dùng đâu
fastapi
uvicorn
jupyter
notebook
pytest  # Có pytest nhưng không có test
```

**Vấn đề:**
- 20 dependencies nhưng chỉ dùng ~10
- `ultralytics`, `transformers`, `timm` không thấy dùng → bloat
- Không pin versions → reproducibility issue

**Giải pháp:**
```toml
# pyproject.toml
[project]
dependencies = [
    "opencv-python>=4.8.0,<5.0.0",
    "numpy>=1.24.0,<2.0.0",
    "easyocr>=1.7.0",
    "pytesseract>=0.3.10",
    # ... only what you actually use
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "pytest-cov>=4.1.0"]
api = ["fastapi>=0.104.0", "uvicorn>=0.24.0"]
```

### 4. **Comments không có giá trị**
```python
# src/extraction/bar_extractor.py
def _is_number(self, text):
    """
    Kiểm tra text có phải là số không
    """
    cleaned = text.replace('.', '').replace(',', '')...
```

**Vấn đề:**
- Comment chỉ lặp lại tên hàm
- Không giải thích WHY, chỉ giải thích WHAT (code đã tự giải thích rồi)

**Better:**
```python
def _is_number(self, text: str) -> bool:
    """Check if text represents a numeric value.
    
    Handles formats: 100, 100.5, 100,000, -50, 75%
    
    Returns:
        True if text is numeric (allows separators and units)
    """
```

### 5. **Type hints thiếu hoặc sai**
```python
# Nhiều hàm không có type hints
def extract(self, image_path, ocr_method='easyocr'):
    ...

# Type hints sai
def _make_categories(num_bars: int, xlabel: str) -> List[str]:
    # Nhưng hàm trả về có thể là generic list
    return [f"Cat{i + 1}" for i in range(num_bars)]
```

**Better:**
```python
from typing import List, Optional, Tuple, Dict, Any

def extract(
    self, 
    image_path: str, 
    ocr_method: str = 'easyocr'
) -> Dict[str, Any]:
    ...

def _make_categories(
    num_bars: int, 
    xlabel: str
) -> List[str]:
    ...
```

### 6. **Xử lý lỗi không đầy đủ**
```python
# generate_dataset.py
try:
    meta = _generate_single_bar_chart(i, num_charts)
    annotations.append(meta)
except Exception as exc:  # noqa: BLE001
    msg = f"Error generating chart {i}: {exc}"
    errors.append(msg)
    print(msg, file=sys.stderr, flush=True)
```

**Vấn đề:**
- Catch `Exception` quá rộng
- `noqa: BLE001` = "tôi biết đây là bad practice nhưng tôi ignore"
- Không log traceback → khó debug

**Better:**
```python
try:
    meta = _generate_single_bar_chart(i, num_charts)
    annotations.append(meta)
except (ValueError, IOError) as exc:
    logger.error(
        f"Failed to generate chart {i}: {exc}", 
        exc_info=True
    )
    errors.append(str(exc))
except Exception as exc:
    logger.critical(
        f"Unexpected error generating chart {i}: {exc}", 
        exc_info=True
    )
    raise  # Re-raise để catch ở outer level
```

---

## 🔧 Vấn đề kỹ thuật cụ thể

### 1. **Performance bottlenecks**

#### **a) O(n²) algorithms không cần thiết**
```python
# chart_detector.py
def _merge_nearby_bars_fallback(self, bars, max_distance):
    # Nested loop: O(n²)
    for i, bar1 in enumerate(bars):
        for j, bar2 in enumerate(bars):
            if j <= i or j in used_indices:
                continue
```

**Đã có optimized version** với KDTree (O(n log n)) nhưng fallback vẫn tệ.

#### **b) Redundant computations**
```python
# OCR được chạy nhiều lần trên cùng regions
for (x, y, box_w, box_h) in text_regions:
    region = image[y:y+box_h, x:x+box_w]
    texts = self.read_text_easyocr(region, confidence_threshold=0.2)
    # Nếu vertical, OCR lại với rotation
    if is_vertical:
        texts = self.read_text_rotated(region, rotation_angles=[0, 90, 270])
```

**Better:** Cache OCR results hoặc batch processing.

### 2. **Data structure không tối ưu**
```python
# Bar representation
bar = {
    'bbox': (x, y, w, h),
    'area': area,
    'center': (cx, cy)
}
```

**Vấn đề:**
- Dict lookup chậm hơn object attributes
- Không type safety
- Không IDE autocomplete

**Better:**
```python
from dataclasses import dataclass

@dataclass
class Bar:
    bbox: Tuple[int, int, int, int]
    area: int
    center: Tuple[int, int]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    @property
    def aspect_ratio(self) -> float:
        return self.height / self.width if self.width > 0 else 0
```

### 3. **Heuristics quá phức tạp và dễ break**

Ví dụ: `_extract_categories()` có hơn 100 lines logic với:
- Pattern matching cho month names
- OCR error correction
- Fuzzy matching
- Dynamic thresholding

**Vấn đề:**
- Khó maintain
- Dễ break với edge cases
- Không scalable

**Better approach:** Machine learning
```python
class CategoryExtractor:
    def __init__(self):
        self.model = load_pretrained_model('category_classifier')
    
    def extract(self, image, bars, text_regions):
        features = self._extract_features(image, bars, text_regions)
        categories = self.model.predict(features)
        return categories
```

---

## 🎯 Recommendations (Ưu tiên cao → thấp)

### **Priority 1: Critical (Làm ngay)**

1. **Add tests**
   ```bash
   tests/
   ├── test_preprocessing/
   │   ├── test_image_utils.py
   │   ├── test_chart_detector.py
   │   └── test_validators.py
   ├── test_extraction/
   │   ├── test_bar_extractor.py
   │   └── test_ocr_engine.py
   └── fixtures/
       └── sample_charts/
   ```
   Target: >80% coverage cho core modules

2. **Refactor monster functions**
   - `BarChartExtractor.extract()`: 100+ lines → split thành 5-6 methods
   - `OCREngine.read_chart_labels()`: 200+ lines → extract helpers
   - `ChartComponentDetector.detect_bars()`: Tương tự

3. **Clean up dependencies**
   - Xóa unused: `ultralytics`, `transformers`, `timm`
   - Pin versions
   - Tách `dev` vs `prod` dependencies

### **Priority 2: Important (Làm trong 1-2 tuần)**

4. **Add type hints đầy đủ**
   - Run `mypy` để check
   - Fix tất cả type errors

5. **Extract magic numbers**
   ```python
   class ChartLayout:
       TITLE_REGION = (0, 0.3)  # top 30%
       YLABEL_REGION = (0, 0.3)  # left 30%
       XLABEL_REGION = (0.85, 1.0)  # bottom 15%
   ```

6. **Improve error messages**
   - Add context: "Failed to detect X-axis for image 'chart_001.png' (size: 800x600)"
   - Include suggestions: "Try increasing hough_threshold or checking image quality"

### **Priority 3: Nice to have**

7. **Add monitoring/metrics**
   ```python
   @dataclass
   class DetectionMetrics:
       num_bars_detected: int
       ocr_confidence: float
       processing_time_ms: float
       warnings: List[str]
   ```

8. **Create visualization tools**
   ```python
   def visualize_detection(image, bars, axes, labels):
       """Draw detected components on image for debugging"""
       ...
   ```

9. **Add configuration file support**
   ```yaml
   # config.yaml
   detector:
     hough_threshold: 100
     min_bar_area: 200
   ocr:
     engine: easyocr
     languages: [en, vi]
     confidence_threshold: 0.5
   ```

---

## 📈 Scoring

| Tiêu chí | Điểm | Ghi chú |
|----------|------|---------|
| **Code Quality** | 4/10 | Nhiều code smell, vi phạm principles |
| **Architecture** | 6/10 | Structure OK nhưng coupling cao |
| **Testing** | 0/10 | Không có test |
| **Documentation** | 3/10 | Comments kém, thiếu docstrings |
| **Performance** | 5/10 | Có optimization nhưng còn bottlenecks |
| **Maintainability** | 4/10 | Khó refactor do thiếu tests |
| **Technical Approach** | 7/10 | Pipeline hợp lý, validators tốt |

**Tổng: 4.1/10** - Dưới trung bình, cần cải thiện nhiều.

---

## 💡 Kết luận

**Điểm tích cực:**
- Ý tưởng và pipeline tốt
- Có validation chain và adaptive config
- Cấu trúc project rõ ràng

**Vấn đề chính:**
- **Code quality thấp**: Functions quá dài, magic numbers, duplicate code
- **Thiếu tests hoàn toàn**: Không thể đảm bảo correctness
- **Dependencies bloated**: Nhiều package không dùng
- **Hard to maintain**: Khó refactor, khó debug

---

## 🔴 BỔ SUNG ĐÁNH GIÁ - PHẦN THẲNG THẮN

### **VẤN ĐỀ LỚN NHẤT: PROJECT CHƯA HOÀN THIỆN - NÓI THẲNG LÀ "NỬA VỜI"**

#### **1. Placeholder files - Code rỗng khắp nơi**

```python
# main.py - ENTRY POINT CHÍNH CỦA PROJECT
if __name__ == "__main__":
    pass  # LITERALLY NOTHING

# src/models/classifier.py
"""Chart type classifier (placeholder)."""
# EMPTY FILE

# src/models/text_generator.py  
"""Text generation model interface (placeholder)."""
# EMPTY FILE

# src/extraction/line_extractor.py
"""Line chart data extraction (placeholder)."""
# EMPTY FILE
```

**Thực tế:** Project chỉ làm được BAR CHART. Line chart, pie chart được quảng cáo trong README nhưng KHÔNG CÓ CODE.

#### **2. README nói dối - Quảng cáo sai sự thật**

README viết:
> "Hybrid approach kết hợp rule-based computer vision algorithms và machine learning models"

**Thực tế:** 
- KHÔNG CÓ ML models nào cả
- `classifier.py` rỗng
- `text_generator.py` rỗng
- Chỉ có pure rule-based CV + OCR
- Đây là **FALSE ADVERTISING** về capabilities của project

README viết:
> "Nhận dạng và phân loại biểu đồ (bar chart, line chart, pie chart)"

**Thực tế:**
- Chỉ xử lý được bar chart
- `line_extractor.py` = placeholder
- Không có `pie_extractor.py`

#### **3. Demo script BỊ BROKEN - Gọi methods không tồn tại**

```python
# scripts/demo_preprocess_bar_charts.py line 78-83
enhanced = preprocessor.preprocess_pipeline(
    str(img_path),
    blur_ksize=(3, 3),        # KHÔNG CÓ parameter này
    apply_blur=True,          # KHÔNG CÓ parameter này
    normalize=False,          # KHÔNG CÓ parameter này
)

# line 105
edges_adaptive = preprocessor.detect_edges_adaptive(enhanced)
# METHOD KHÔNG TỒN TẠI trong ImagePreprocessor class
```

**Thực tế:** Script này sẽ CRASH ngay khi chạy vì gọi methods không tồn tại.

**Điều này cho thấy:**
- Không ai test code
- Code được viết rồi bỏ đó
- Thiếu integration testing

---

### **VẤN ĐỀ KỸ THUẬT NGHIÊM TRỌNG**

#### **4. Không có input validation ở entry points**

```python
# bar_extractor.py
def extract(self, image_path, ocr_method='easyocr'):
    image = cv2.imread(image_path)  # Nếu file không tồn tại?
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CRASH nếu image=None
```

**Vấn đề:**
- Không check file tồn tại
- Không validate image format
- Không handle corrupted images
- User nhận được cryptic OpenCV error thay vì helpful message

**Phải là:**
```python
def extract(self, image_path: str, ocr_method: str = 'easyocr') -> Dict[str, Any]:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidImageError(f"Failed to load image: {image_path}")
    
    if len(image.shape) != 3:
        raise InvalidImageError(f"Expected color image, got shape: {image.shape}")
```

#### **5. Silent failures khắp nơi**

```python
# ocr_engine.py
def read_text_rotated(self, image_region, rotation_angles=[0, 90, 270]):
    best_result = None
    best_confidence = 0
    
    for angle in rotation_angles:
        # ... xử lý ...
        texts = self.read_text_easyocr(rotated, confidence_threshold=0.2)
        
        if texts and texts[0]['confidence'] > best_confidence:
            best_confidence = texts[0]['confidence']
            best_result = texts[0]
    
    return [best_result] if best_result else []  # Trả về [] nếu fail
```

**Vấn đề:** 
- OCR fail silently
- Không biết tại sao không đọc được text
- Không có retry logic
- Không có fallback mechanism đúng cách

#### **6. Mutable default arguments - Python anti-pattern**

```python
# ocr_engine.py line 68
def read_text_rotated(self, image_region, rotation_angles=[0, 90, 270]):
#                                         ^^^^^^^^^^^^^^^^^^^^^^^^
# MUTABLE DEFAULT ARGUMENT - Classic Python bug
```

**Phải là:**
```python
def read_text_rotated(
    self, 
    image_region, 
    rotation_angles: Optional[List[int]] = None
):
    if rotation_angles is None:
        rotation_angles = [0, 90, 270]
```

#### **7. Global state và side effects không kiểm soát**

```python
# generate_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAR_CHART_DIR = PROJECT_ROOT / "data" / "raw" / "bar_charts"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
```

**Vấn đề:**
- Global constants nhưng depend on file location
- Không thể override cho testing
- Hardcoded paths không portable

---

### **VẤN ĐỀ THIẾT KẾ**

#### **8. Tight coupling giữa các modules**

```python
# bar_extractor.py
class BarChartExtractor:
    def __init__(self):
        self.detector = ChartComponentDetector()  # Hardcoded dependency
        self.ocr = OCREngine()                     # Hardcoded dependency
```

**Vấn đề:**
- Không thể inject mock dependencies cho testing
- Không thể swap implementations
- Khó extend

**Phải là:**
```python
class BarChartExtractor:
    def __init__(
        self, 
        detector: Optional[ChartComponentDetector] = None,
        ocr: Optional[OCREngine] = None
    ):
        self.detector = detector or ChartComponentDetector()
        self.ocr = ocr or OCREngine()
```

#### **9. Không có abstraction layer**

Tất cả các extractor nên implement chung interface:

```python
# KHÔNG CÓ trong code hiện tại
from abc import ABC, abstractmethod

class ChartExtractor(ABC):
    @abstractmethod
    def extract(self, image_path: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def validate_output(self, result: Dict[str, Any]) -> bool:
        pass

class BarChartExtractor(ChartExtractor):
    def extract(self, image_path: str) -> Dict[str, Any]:
        # implementation
        
class LineChartExtractor(ChartExtractor):
    def extract(self, image_path: str) -> Dict[str, Any]:
        # implementation
```

#### **10. No separation between detection and extraction**

`detect_bars()` trong `chart_detector.py` làm quá nhiều việc:
- Detect bars (OK)
- Filter bars (nên tách)
- Merge bars (nên tách)
- Validate bars (đã tách nhưng gọi inline)

---

### **VẤN ĐỀ VỀ ENGINEERING PRACTICES**

#### **11. Không có CI/CD**
- Không có `.github/workflows/`
- Không có pre-commit hooks
- Không có linting automation
- Không có automated testing

#### **12. Không có benchmarking/evaluation**
- Không biết accuracy là bao nhiêu
- Không có ground truth comparison
- Không có metrics tracking
- Không thể đo lường improvements

#### **13. Mixed language - Inconsistent**
```python
# Đôi khi Vietnamese
def _extract_ylabel_from_left_region(self, image):
    """
    Crop vùng bên trái và OCR riêng để tìm y-label
    Đặc biệt hữu ích cho text dọc (rotated 90 degrees)
    """

# Đôi khi English
def detect_axes(self, image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect X and Y axes using Hough Line Transform with validation.
    """
```

**Chọn một ngôn ngữ và stick với nó** - preferably English cho code.

#### **14. Notebooks thay cho tests**

Hiện tại chỉ có notebooks để "test":
- `test_10_charts.ipynb`
- `test_10_charts_executed.ipynb`

**Vấn đề:**
- Notebooks không tự động chạy
- Không có assertions
- Không có CI integration
- Kết quả phụ thuộc vào người chạy

---

### **VẤN ĐỀ VỀ SCALABILITY**

#### **15. Heuristics không scalable**

```python
# bar_extractor.py - 50+ lines chỉ để match month names và fix OCR errors
month_patterns = {
    'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', ...
    'jui': 'Jul', 'ju1': 'Jul', 'juI': 'Jul', ...  # OCR fixes
}
```

**Vấn đề:**
- Không scale cho nhiều ngôn ngữ
- Không handle edge cases mới
- Cần manual update cho mỗi pattern mới
- Đây là việc ML nên làm, không phải hardcoded rules

#### **16. Memory usage không được quản lý**

```python
# Không có image cleanup
image = cv2.imread(image_path)
# ... xử lý ...
# Không có del image hoặc explicit cleanup
```

Với batch processing nhiều images, đây sẽ là memory leak.

---

## 📊 ĐÁNH GIÁ LẠI (THỰC TẾ HƠN)

| Tiêu chí | Điểm | Lý do |
|----------|------|-------|
| **Completeness** | 2/10 | Chỉ làm được bar chart, 60% code là placeholder |
| **Code Quality** | 3/10 | Magic numbers, no validation, broken scripts |
| **Testing** | 0/10 | Literally zero tests |
| **Documentation** | 2/10 | README nói dối, comments không helpful |
| **Production Ready** | 1/10 | Không thể deploy được |
| **Maintainability** | 3/10 | Tight coupling, no abstractions |
| **Engineering** | 2/10 | No CI/CD, no benchmarks, no metrics |

**Tổng thực tế: 1.9/10**

---

## 🎯 PHẢI LÀM GÌ ĐỂ PROJECT NÀY USABLE

### **Immediate (Làm ngay hoặc đừng claim là "project")**

1. **Sửa README** - Bỏ hết những gì không có:
   - Xóa "hybrid approach với ML"
   - Xóa "line chart, pie chart support"
   - Viết rõ: "Currently only supports bar charts with rule-based approach"

2. **Fix broken demo script** hoặc XÓA nó đi

3. **Implement main.py** với actual functionality:
   ```python
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument('image', help='Path to chart image')
       parser.add_argument('--output', '-o', help='Output JSON file')
       args = parser.parse_args()
       
       extractor = BarChartExtractor()
       result = extractor.extract(args.image)
       
       if args.output:
           with open(args.output, 'w') as f:
               json.dump(result, f, indent=2)
       else:
           print(json.dumps(result, indent=2))
   ```

4. **Viết ít nhất 10 unit tests** cho core functionality

### **Short-term (1-2 tuần)**

5. Add input validation EVERYWHERE
6. Fix mutable default arguments
7. Implement dependency injection
8. Add proper error messages
9. Pin all dependency versions

### **Medium-term (1 tháng)**

10. Implement actual ML classifier (nếu muốn claim "hybrid")
11. Add line chart support (hoặc xóa khỏi scope)
12. Add benchmarking framework
13. Add CI/CD pipeline

---

## 💀 KẾT LUẬN THẲNG THẮN

**Project này ở trạng thái "bỏ dở giữa chừng":**

1. **README over-promises, code under-delivers** - Đây là red flag lớn nhất
2. **Không thể chạy được out-of-the-box** - Demo script broken
3. **Không có entry point** - main.py rỗng
4. **Không có tests** - Không ai biết code có work không
5. **60% features là placeholder** - Line chart, pie chart, classifier, text generator đều rỗng

**Nếu đây là:**
- **Homework/Learning project:** Cần làm lại từ đầu với scope nhỏ hơn, focus vào bar chart cho xong rồi mở rộng
- **Production project:** KHÔNG READY. Cần 2-3 tháng để đưa vào trạng thái usable
- **Portfolio project:** Cần clean up và honest về capabilities

**Lời khuyên:** 
- **Đừng thêm features mới** cho đến khi bar chart extraction hoàn thiện và có tests
- **Thu nhỏ scope** - làm 1 thứ cho tốt còn hơn làm 5 thứ dở dang
- **Viết tests trước** khi code thêm bất cứ gì

---

# 📋 IMPLEMENTATION PLAN CHI TIẾT

> **Nguyên tắc:** Plan được thiết kế để có thể **dừng ở bất kỳ phase nào** và project vẫn ở trạng thái hoạt động được.

---

## [V] **PHASE 0: EMERGENCY FIXES (1-2 ngày)**

*Mục tiêu: Project có thể chạy được, không nói dối trong README*

### **Task 0.1: Sửa README.md - Honest về capabilities**

**File:** `README.md`

**Thay đổi:**
- Xóa "Hybrid approach kết hợp rule-based CV và ML models" → "Rule-based computer vision + OCR approach"
- Xóa "Nhận dạng bar chart, line chart, pie chart" → "Hiện tại chỉ hỗ trợ bar chart (vertical)"
- Thêm note: "Line chart, pie chart: planned cho phiên bản sau"

---

### **Task 0.2: Implement main.py**

**File:** `main.py`

**Nội dung:**
```python
"""
Chart Understanding - Bar Chart Data Extraction

Usage:
    python main.py <image_path> [--output output.json]
    python main.py data/raw/bar_charts/chart_0001.png
    python main.py chart.png -o result.json
"""
import argparse
import json
import sys
from pathlib import Path

from src.extraction.bar_extractor import BarChartExtractor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract data from bar chart images"
    )
    parser.add_argument("image", type=str, help="Path to bar chart image")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON file path (default: print to stdout)")
    parser.add_argument("--ocr", type=str, choices=["easyocr", "tesseract"],
                        default="easyocr", help="OCR engine to use")
    
    args = parser.parse_args()
    
    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1
    
    # Extract data
    try:
        extractor = BarChartExtractor()
        result = extractor.extract(str(image_path), ocr_method=args.ocr)
    except Exception as e:
        print(f"Error: Failed to extract data: {e}", file=sys.stderr)
        return 1
    
    # Output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    
    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Result saved to: {args.output}")
    else:
        print(output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### **Task 0.3: Fix hoặc xóa demo script broken**

**File:** `scripts/demo_preprocess_bar_charts.py`

**Option A (Recommended):** Xóa file nếu không cần

**Option B:** Fix các dòng lỗi:
- Line 78-83: Bỏ parameters không tồn tại (`blur_ksize`, `apply_blur`, `normalize`)
- Line 105: Đổi `detect_edges_adaptive()` thành `detect_edges()`

---

### **Task 0.4: Clean requirements.txt**

**File:** `requirements.txt`

**Xóa dependencies không dùng:**
- `torch`, `torchvision` (no ML model)
- `transformers`, `timm` (no transformer)
- `ultralytics` (no YOLO)
- `scikit-learn`, `scikit-image` (not actively used)
- `fastapi`, `uvicorn` (no API implemented)

**Pin versions cho các dependencies còn lại.**

**Tạo thêm:** `requirements-dev.txt` cho dev dependencies (pytest, mypy, ruff)

---

## [] **PHASE 1: FOUNDATION (1 tuần)**

*Mục tiêu: Có tests, input validation, code có thể maintain được*

### **Task 1.1: Tạo cấu trúc tests**

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── fixtures/
│   └── sample_charts/
│       ├── simple_bar.png
│       └── expected/
│           └── simple_bar.json
├── test_preprocessing/
│   ├── __init__.py
│   ├── test_image_utils.py
│   ├── test_chart_detector.py
│   └── test_validators.py
└── test_extraction/
    ├── __init__.py
    ├── test_bar_extractor.py
    └── test_ocr_engine.py
```

---

### **Task 1.2: Viết conftest.py với fixtures**

**File:** `tests/conftest.py`

Tạo fixtures cho:
- `sample_bar_image`: Load hoặc tạo dummy bar chart image
- `expected_simple_bar`: Expected output structure
- `mock_bars`: Sample bar detections cho validator testing
- `mock_bars_with_outlier`: Bars với outlier để test filtering

---

### **Task 1.3: Viết tests cho validators (QUAN TRỌNG NHẤT)**

**File:** `tests/test_preprocessing/test_validators.py`

**Test cases:**
- `TestWidthValidator`: Test width consistency filtering
- `TestAreaValidator`: Test tiny noise removal
- `TestSpacingValidator`: Test even spacing detection
- `TestValidationPipeline`: Test chain of validators

---

### **Task 1.4: Viết tests cho chart_detector**

**File:** `tests/test_preprocessing/test_chart_detector.py`

**Test cases:**
- `test_validate_image_rejects_none`
- `test_validate_image_rejects_wrong_type`
- `test_validate_image_accepts_rgb`
- `test_detect_axes_returns_tuple`
- `test_detect_bars_returns_list`

---

### **Task 1.5: Add input validation to bar_extractor.py**

**File:** `src/extraction/bar_extractor.py`

Thêm vào đầu method `extract()`:
- Check file exists
- Validate image format (png, jpg, etc.)
- Validate ocr_method value
- Handle cv2.imread() returning None

---

### **Task 1.6: Fix mutable default argument in ocr_engine.py**

**File:** `src/extraction/ocr_engine.py`

```python
# TRƯỚC (BUG)
def read_text_rotated(self, image_region, rotation_angles=[0, 90, 270]):

# SAU (FIXED)
def read_text_rotated(self, image_region, rotation_angles=None):
    if rotation_angles is None:
        rotation_angles = [0, 90, 270]
```

---

### **Task 1.7: Add pytest.ini**

**File:** `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

---

## [] **PHASE 2: CODE QUALITY (1-2 tuần)**

*Mục tiêu: Refactor code, add type hints, extract magic numbers*

### **Task 2.1: Tạo dataclasses cho data structures**

**File mới:** `src/models/data_types.py`

Tạo các classes:
- `BoundingBox`: Rectangle với properties (x2, y2, center, area)
- `Bar`: Detected bar với bbox, color, properties (width, height, aspect_ratio)
- `TextRegion`: OCR result với bbox, text, confidence
- `ChartLabels`: title, x_label, y_label, categories, values
- `BarChartData`: Structured output

---

### **Task 2.2: Extract magic numbers vào config**

**File mới:** `src/config/layout_config.py`

```python
@dataclass(frozen=True)
class ChartLayoutConfig:
    TITLE_REGION_TOP: float = 0.0
    TITLE_REGION_BOTTOM: float = 0.3  # Top 30%
    YLABEL_REGION_LEFT: float = 0.0
    YLABEL_REGION_RIGHT: float = 0.15  # Left 15%
    XLABEL_REGION_TOP: float = 0.85  # Bottom 15%
    # ... etc
```

---

### **Task 2.3: Refactor bar_extractor.py - Split monster function**

**File:** `src/extraction/bar_extractor.py`

Split `extract()` thành:
- `_load_and_validate_image()`
- `_detect_components()`
- `_extract_labels()`
- `_calculate_values()`
- `_extract_categories()`
- `_build_output()`

Add dependency injection trong `__init__()`.

---

### **Task 2.4: Add type hints toàn bộ**

Chạy `mypy src/ --ignore-missing-imports` và fix tất cả errors.

---

## [] **PHASE 3: ENGINEERING (2-3 tuần)**

*Mục tiêu: CI/CD, benchmarking, documentation*

### **Task 3.1: Add GitHub Actions CI**

**File:** `.github/workflows/ci.yml`

Jobs:
- Install dependencies
- Run linting (ruff)
- Run type checking (mypy)
- Run tests với coverage
- Upload coverage report

---

### **Task 3.2: Add benchmarking script**

**File:** `scripts/benchmark.py`

Features:
- Load annotated dataset
- Run extraction on each image
- Compare with ground truth
- Calculate metrics: bar count accuracy, value accuracy, processing time
- Print summary report

---

### **Task 3.3: Add pre-commit hooks**

**File:** `.pre-commit-config.yaml`

Hooks:
- ruff (linting + formatting)
- mypy (type checking)
- trailing-whitespace
- end-of-file-fixer
- check-yaml

---

## [] **PHASE 4: OPTIONAL FEATURES (Future)**

*Chỉ làm khi Phase 0-3 hoàn thành*

- Task 4.1: Implement line chart extractor
- Task 4.2: Add ML-based chart classifier
- Task 4.3: Add REST API với FastAPI
- Task 4.4: Add visualization tools cho debugging

---

## 📊 SUMMARY - THỨ TỰ THỰC HIỆN

| Phase | Tasks | Thời gian | Output |
|-------|-------|-----------|--------|
| **Phase 0** | 0.1-0.4 | 1-2 ngày | Project chạy được, README trung thực |
| **Phase 1** | 1.1-1.7 | 1 tuần | Có tests, input validation |
| **Phase 2** | 2.1-2.4 | 1-2 tuần | Code refactored, type hints |
| **Phase 3** | 3.1-3.3 | 2-3 tuần | CI/CD, benchmarking |
| **Phase 4** | 4.1-4.4 | Ongoing | New features |

---

## ✅ CHECKLIST THEO DÕI TIẾN ĐỘ

### Phase 0: Emergency Fixes
- [ ] Task 0.1: Sửa README.md
- [ ] Task 0.2: Implement main.py
- [ ] Task 0.3: Fix/xóa demo script broken
- [ ] Task 0.4: Clean requirements.txt

### Phase 1: Foundation
- [ ] Task 1.1: Tạo cấu trúc tests
- [ ] Task 1.2: Viết conftest.py
- [ ] Task 1.3: Tests cho validators
- [ ] Task 1.4: Tests cho chart_detector
- [ ] Task 1.5: Input validation cho bar_extractor
- [ ] Task 1.6: Fix mutable default arguments
- [ ] Task 1.7: Add pytest.ini

### Phase 2: Code Quality
- [ ] Task 2.1: Tạo dataclasses
- [ ] Task 2.2: Extract magic numbers
- [ ] Task 2.3: Refactor bar_extractor
- [ ] Task 2.4: Add type hints

### Phase 3: Engineering
- [ ] Task 3.1: GitHub Actions CI
- [ ] Task 3.2: Benchmarking script
- [ ] Task 3.3: Pre-commit hooks

### Phase 4: Optional Features
- [ ] Task 4.1: Line chart extractor
- [ ] Task 4.2: ML classifier
- [ ] Task 4.3: REST API
- [ ] Task 4.4: Visualization tools