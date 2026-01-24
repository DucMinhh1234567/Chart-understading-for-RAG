# src/preprocessing/image_utils.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        pass
    
    def load_image(self, image_path):
        """Load và convert sang RGB"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def remove_noise(self, image):
        """Giảm noise"""
        # Gaussian blur
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised
    
    def enhance_contrast(self, image):
        """Tăng contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def binarize(self, image, method='otsu'):
        """Chuyển sang ảnh binary"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        return binary
    
    def detect_edges(self, image):
        """Phát hiện cạnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def preprocess_pipeline(self, image_path):
        """Full preprocessing pipeline"""
        img = self.load_image(image_path)
        img = self.remove_noise(img)
        img = self.enhance_contrast(img)
        return img
    
    def enhance_for_ocr(self, image):
        """
        CẢI THIỆN: Cải thiện image quality cho OCR
        Tăng contrast, denoise và sharpen để OCR chính xác hơn
        """
        # 1. Tăng contrast với CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Denoise để giảm noise
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced, None, 10, 10, 7, 21
        )
        
        # 3. Sharpen để làm rõ text (tùy chọn, có thể bỏ qua nếu làm mờ quá)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def preprocess_for_ocr(self, image_path):
        """
        CẢI THIỆN: Pipeline preprocessing tối ưu cho OCR
        """
        img = self.load_image(image_path)
        img = self.enhance_for_ocr(img)
        return img