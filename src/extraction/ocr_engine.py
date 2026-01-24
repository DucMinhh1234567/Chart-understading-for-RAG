# src/extraction/ocr_engine.py
import easyocr
import pytesseract
from PIL import Image
import numpy as np

class OCREngine:
    def __init__(self, languages=None, gpu=False):
        """
        Initialize OCR engines
        """
        if languages is None:
            languages = ['en', 'vi']
        # EasyOCR - tốt cho tiếng Việt
        self.easyocr_reader = easyocr.Reader(languages, gpu=gpu)
        
        # Tesseract - backup
        # Cần cài: sudo apt-get install tesseract-ocr tesseract-ocr-vie
    
    def read_text_easyocr(self, image_region, confidence_threshold=0.5):
        """
        Đọc text từ image region bằng EasyOCR
        """
        results = self.easyocr_reader.readtext(image_region)
        
        # results format: [([box], text, confidence), ...]
        texts = []
        for (bbox, text, conf) in results:
            if conf > confidence_threshold:
                texts.append({
                    'text': text.strip(),
                    'confidence': conf,
                    'bbox': bbox
                })
        
        return texts
    
    def read_text_tesseract(self, image_region):
        """
        Đọc text bằng Tesseract
        """
        # Convert numpy array to PIL Image
        if isinstance(image_region, np.ndarray):
            image_region = Image.fromarray(image_region)
        
        # OCR
        text = pytesseract.image_to_string(image_region, lang='vie+eng')
        
        # Get detailed data
        data = pytesseract.image_to_data(image_region, lang='vie+eng',
                                        output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 50:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    results.append({
                        'text': text,
                        'confidence': data['conf'][i] / 100,
                        'bbox': (data['left'][i], data['top'][i],
                                data['width'][i], data['height'][i])
                    })
        
        return results
    
    def read_text_rotated(self, image_region, rotation_angles=[0, 90, 270]):
        """
        Đọc text với nhiều góc rotation để bắt được text dọc (y-axis labels)
        Thử nhiều góc xoay và chọn kết quả tốt nhất
        
        Args:
            image_region: Image region để OCR
            rotation_angles: Danh sách các góc xoay (0, 90, 270)
        
        Returns:
            List of text results với confidence cao nhất
        """
        import cv2
        
        best_result = None
        best_confidence = 0
        
        for angle in rotation_angles:
            # Rotate region
            if angle == 90:
                rotated = cv2.rotate(image_region, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 270:
                rotated = cv2.rotate(image_region, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image_region, cv2.ROTATE_180)
            else:
                rotated = image_region
            
            # OCR on rotated image
            texts = self.read_text_easyocr(rotated, confidence_threshold=0.2)
            
            if texts and texts[0]['confidence'] > best_confidence:
                best_confidence = texts[0]['confidence']
                best_result = texts[0]
        
        return [best_result] if best_result else []
    
    def _is_number(self, text):
        """
        Kiểm tra text có phải là số không
        """
        # Loại bỏ dấu chấm, dấu phẩy, dấu trừ, dấu phần trăm
        cleaned = text.replace('.', '').replace(',', '').replace('-', '').replace('%', '').strip()
        
        # Kiểm tra có phải số nguyên hoặc số thập phân
        try:
            float(cleaned)
            return True
        except ValueError:
            # Kiểm tra có phải số với đơn vị (ví dụ: "100%", "50k")
            if cleaned and cleaned[0].isdigit():
                return True
            return False
    
    def read_chart_labels(self, image, text_regions, ocr_method='easyocr'):
        """
        Đọc các labels trong biểu đồ với improved classification logic
        Sử dụng heuristics để phân biệt title, labels và ticks
        """
        labels = {
            'title': None,
            'xlabel': None,
            'ylabel': None,
            'legend': [],
            'values': []
        }
        
        h, w = image.shape[:2]
        all_texts = []  # Lưu tất cả texts để phân tích
        
        # Bước 1: Đọc tất cả texts với confidence threshold thấp hơn
        for (x, y, box_w, box_h) in text_regions:
            # Crop region
            region = image[y:y+box_h, x:x+box_w]
            
            # CẢI THIỆN: Tính aspect ratio để detect rotated text (vertical text có height > width)
            aspect_ratio = box_h / box_w if box_w > 0 else 1.0
            is_vertical = aspect_ratio > 1.5  # Text dọc nếu height > 1.5 * width
            
            # Nếu là vertical text, thử rotate để OCR tốt hơn
            if is_vertical:
                texts = self.read_text_rotated(region, rotation_angles=[0, 90, 270])
            else:
                # OCR với confidence threshold thấp hơn để bắt được nhiều text hơn
                texts = self.read_text_easyocr(region, confidence_threshold=0.2)
            
            if not texts:
                continue
            
            text = texts[0]['text']
            center_y = y + box_h // 2
            center_x = x + box_w // 2
            
            all_texts.append({
                'text': text,
                'position': (center_x, center_y),
                'bbox': (x, y, box_w, box_h),
                'length': len(text),
                'is_number': self._is_number(text),
                'font_size': box_h,  # Ước tính font size từ height
                'aspect_ratio': aspect_ratio,
                'is_vertical': is_vertical  # Flag để identify vertical text (y-axis labels)
            })
        
        # Bước 2: Phân loại với heuristics (CẢI THIỆN)
        
        # ========== CẢI THIỆN TITLE EXTRACTION ==========
        # Tìm tất cả texts ở vùng top (y < 0.3 * h)
        title_region_texts = [
            t for t in all_texts 
            if t['position'][1] < 0.3 * h and not t['is_number']
        ]
        
        if title_region_texts:
            # CẢI THIỆN: Group texts theo dòng (cùng y level) trước, sau đó sort theo x
            # Để đảm bảo thứ tự đọc đúng (trái sang phải) - tránh "Sales Monthly" thay vì "Monthly Sales"
            title_lines = {}  # Group by y position (rounded)
            
            for text_info in title_region_texts:
                y_pos = text_info['position'][1]
                y_line = round(y_pos / 20) * 20  # Round to nearest 20 pixels (same line)
                
                if y_line not in title_lines:
                    title_lines[y_line] = []
                title_lines[y_line].append(text_info)
            
            # Sort các dòng theo y (từ trên xuống)
            sorted_lines = sorted(title_lines.items())
            
            # Kết hợp texts từ mỗi dòng (sort theo x trong mỗi dòng)
            title_parts = []
            for y_line, texts_in_line in sorted_lines:
                # Sort theo x position (trái sang phải) trong cùng dòng
                texts_in_line.sort(key=lambda t: t['position'][0])
                
                # Kết hợp texts trong cùng dòng
                line_text = ' '.join([t['text'] for t in texts_in_line])
                title_parts.append(line_text)
            
            if title_parts:
                # Kết hợp các dòng thành title
                full_title = ' '.join(title_parts)
                # Chọn title dài nhất và có font size lớn nhất
                if not labels['title'] or len(full_title) > len(labels['title']):
                    labels['title'] = full_title
            
            # Fallback: nếu không kết hợp được, chọn text dài nhất
            if not labels['title'] and title_region_texts:
                title_region_texts.sort(key=lambda t: (-t['font_size'], -t['length']))
                labels['title'] = title_region_texts[0]['text']
        
        # ========== CẢI THIỆN Y-AXIS LABEL EXTRACTION ==========
        # Y-axis label thường là text DỌC (rotated 90°) ở bên trái
        # Ưu tiên tìm text có aspect_ratio > 1.5 (height > width)
        ylabel_candidates = []
        
        for t in all_texts:
            x_pos, y_pos = t['position']
            
            # Vùng bên trái (x < 0.3 * w để bao gồm cả rotated text)
            if (x_pos < 0.3 * w and 
                y_pos < 0.85 * h and  # Mở rộng vùng tìm kiếm
                y_pos > 0.15 * h and  # Không quá gần top
                not t['is_number'] and
                t['length'] > 0):  # Cho phép cả text 1 ký tự
            
                # CẢI THIỆN: Ưu tiên text dọc (rotated 90°) - đây là y-axis label
                # Y-axis label thường có height > width (aspect_ratio > 1.5)
                is_vertical_text = t.get('is_vertical', False) or t.get('aspect_ratio', 1.0) > 1.5
                
                # Tính điểm ưu tiên
                priority_score = 0
                
                # Ưu tiên cao cho vertical text (y-axis label thường dọc)
                if is_vertical_text:
                    priority_score += 100
                
                # Ưu tiên text dài (y-label thường là từ hoặc cụm từ)
                priority_score += min(t['length'] * 10, 50)
                
                # Ưu tiên text ở giữa chiều cao chart (0.3h - 0.6h)
                y_center_score = 1.0 / (abs(y_pos - 0.45 * h) + 1)
                priority_score += y_center_score * 20
                
                # Ưu tiên text không quá gần top (không phải title)
                if y_pos > 0.25 * h:
                    priority_score += 10
                
                ylabel_candidates.append({
                    'text_info': t,
                    'priority': priority_score,
                    'is_vertical': is_vertical_text
                })
        
        if ylabel_candidates:
            # Sort theo priority score
            ylabel_candidates.sort(key=lambda c: -c['priority'])
            
            # Chọn candidate tốt nhất
            best_candidate = ylabel_candidates[0]
            best_ylabel = best_candidate['text_info']
            
            # Kiểm tra thêm: không phải là category label (thường ở dưới)
            if best_ylabel['position'][1] < 0.75 * h:
                labels['ylabel'] = best_ylabel['text']
        
        # X-AXIS LABEL: Text ở bottom, không phải số, không phải category
        xlabel_candidates = [
            t for t in all_texts
            if (t['position'][1] > 0.85 * h and
                t['position'][0] > 0.3 * w and
                not t['is_number'] and
                t['length'] > 1)
        ]
        if xlabel_candidates:
            # Chọn text dài nhất, ưu tiên text ở giữa chiều rộng
            xlabel_candidates.sort(key=lambda t: (
                -t['length'],
                abs(t['position'][0] - 0.5 * w)
            ))
            labels['xlabel'] = xlabel_candidates[0]['text']
        
        # VALUES: Tất cả texts còn lại (có thể là values trên bars, ticks, categories)
        for t in all_texts:
            # Bỏ qua các texts đã được classify
            if (t['text'] == labels.get('title') or
                t['text'] == labels.get('xlabel') or
                t['text'] == labels.get('ylabel')):
                continue
            
            labels['values'].append({
                'text': t['text'],
                'position': t['position'],
                'is_number': t['is_number']
            })
        
        return labels