# Chart Understanding Project

## Mô tả Project

Project "chart-understanding" là một hệ thống trích xuất dữ liệu từ biểu đồ cột (bar chart) tự động. Project sử dụng rule-based computer vision + OCR approach để phát hiện các thành phần biểu đồ và đọc text/giá trị.

**Trạng thái hiện tại:** Chỉ hỗ trợ **vertical bar chart**. Line chart và pie chart đang được phát triển.

## Tính năng

- Trích xuất dữ liệu từ bar chart (vertical)
- Phát hiện trục X, Y và các thanh (bars)
- Sử dụng OCR (EasyOCR/Tesseract) để đọc nhãn và giá trị
- Xuất dữ liệu dạng JSON
- Validation pipeline để lọc false positives

## Limitations

- Chỉ hỗ trợ vertical bar chart
- Chưa hỗ trợ: line chart, pie chart, stacked bar, horizontal bar
- Chưa có ML-based chart classifier
- Độ chính xác phụ thuộc vào chất lượng ảnh và độ rõ của text

## Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- pip hoặc conda

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài đặt Tesseract OCR (cho pytesseract)

**Windows:**
- Tải và cài đặt từ: https://github.com/UB-Mannheim/tesseract/wiki
- Thêm đường dẫn vào biến môi trường PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Hướng dẫn sử dụng

### Chạy chương trình chính

```bash
# Cơ bản - in kết quả ra stdout
python main.py <đường_dẫn_ảnh>

# Ví dụ
python main.py data/raw/bar_charts/chart_0001.png

# Lưu kết quả vào file JSON
python main.py chart.png -o result.json

# Sử dụng Tesseract thay vì EasyOCR
python main.py chart.png --ocr tesseract
```

### Output mẫu

```json
{
  "chart_type": "bar_chart",
  "title": "Monthly Sales",
  "x_axis_label": "Months",
  "y_axis_label": "Sales ($)",
  "data": [
    {"category": "Jan", "value": 45.5},
    {"category": "Feb", "value": 62.3},
    {"category": "Mar", "value": 78.1}
  ]
}
```

### Sử dụng các module riêng lẻ

```python
from src.extraction.bar_extractor import BarChartExtractor

extractor = BarChartExtractor()
result = extractor.extract("chart.png", ocr_method="easyocr")
print(result)
```

## Kiến trúc hệ thống

```
chart-understanding/
├── data/
│   ├── raw/
│   │   └── bar_charts/
│   └── annotations/
├── src/
│   ├── preprocessing/
│   │   ├── image_utils.py      # Tiền xử lý ảnh
│   │   ├── chart_detector.py   # Phát hiện axes, bars
│   │   ├── detector_config.py  # Adaptive configuration
│   │   └── bar_validators.py   # Validation pipeline
│   ├── extraction/
│   │   ├── bar_extractor.py    # Main extraction logic
│   │   └── ocr_engine.py       # EasyOCR + Tesseract
│   └── utils/
│       └── visualization.py
├── notebooks/                   # Demo notebooks
├── scripts/
│   └── generate_dataset.py     # Tạo synthetic dataset
├── main.py                      # CLI entry point
└── requirements.txt
```

### Luồng xử lý

1. **Input**: Ảnh bar chart (PNG/JPG)
2. **Preprocessing**: Load ảnh → Enhance contrast → Detect edges
3. **Detection**: Phát hiện trục X, Y → Phát hiện bars → Validation
4. **OCR**: Đọc title, labels, values
5. **Output**: Structured JSON data

## Roadmap phát triển

### Đã hoàn thành
- Preprocessing module
- Bar chart detector với adaptive thresholds
- Validation pipeline (Width, Area, Spacing validators)
- OCR integration (EasyOCR + Tesseract)
- CLI interface

### Đang phát triển
- Line chart extractor
- Cải thiện độ chính xác OCR
- Unit tests

### Kế hoạch tương lai
- Pie chart extractor
- ML-based chart classifier
- REST API
- Horizontal bar chart support
