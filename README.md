# Chart Understanding Project

## Mô tả Project

Project "chart-understanding" là một hệ thống nhận dạng và mô tả biểu đồ tự động, có khả năng phân tích các loại biểu đồ phổ biến (bar chart, line chart, pie chart) và chuyển đổi chúng thành dữ liệu text và JSON. Project sử dụng hybrid approach kết hợp rule-based computer vision algorithms và machine learning models để đạt độ chính xác cao.

## Tính năng

- Nhận dạng và phân loại biểu đồ (bar chart, line chart, pie chart)
- Trích xuất dữ liệu từ biểu đồ
- Chuyển đổi dữ liệu thành định dạng text và JSON
- Sử dụng OCR để đọc nhãn và giá trị trên biểu đồ
- Hybrid approach: kết hợp rule-based và ML models

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
python main.py
```

### Sử dụng các module riêng lẻ

```python
from src.preprocessing import chart_detector
from src.extraction import bar_extractor, line_extractor
from src.models import classifier, text_generator
```

## Kiến trúc hệ thống

```
chart-understanding/
├── data/
│   ├── raw/
│   │   ├── bar_charts/
│   │   ├── line_charts/
│   │   └── pie_charts/
│   ├── processed/
│   └── annotations/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── chart_detector.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── bar_extractor.py
│   │   ├── line_extractor.py
│   │   └── ocr_engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── text_generator.py
│   ├── text_generation/
│   │   ├── __init__.py
│   │   └── template_generator.py
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── notebooks/
├── tests/
├── models/
├── config/
├── scripts/
│   ├── generate_dataset.py
│   └── train_classifier.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Luồng xử lý

1. **Preprocessing**: Nhận ảnh đầu vào → Tiền xử lý → Phát hiện biểu đồ
2. **Classification**: Phân loại loại biểu đồ (bar/line/pie)
3. **Extraction**: Trích xuất dữ liệu dựa trên loại biểu đồ
4. **OCR**: Đọc nhãn và giá trị từ biểu đồ
5. **Text Generation**: Tạo mô tả text và JSON từ dữ liệu đã trích xuất

## Roadmap phát triển

### Phase 1: Foundation (Hiện tại)
- [x] Thiết lập cấu trúc project
- [ ] Implement preprocessing module
- [ ] Implement chart detector
- [ ] Implement basic extraction cho từng loại biểu đồ

### Phase 2: ML Integration
- [ ] Train và tích hợp chart classifier
- [ ] Implement OCR engine với EasyOCR/Pytesseract
- [ ] Tích hợp text generation models

### Phase 3: Enhancement
- [ ] Cải thiện độ chính xác extraction
- [ ] Hỗ trợ thêm các loại biểu đồ (scatter, area, etc.)
- [ ] Tối ưu hóa performance

### Phase 4: Production
- [ ] Tạo API với FastAPI
- [ ] Deploy và testing
- [ ] Documentation hoàn chỉnh
