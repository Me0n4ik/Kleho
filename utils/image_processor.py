import cv2
import numpy as np
import pytesseract
from PIL import Image

def process_image(image):
    """Предобработка изображения для улучшения распознавания"""
    try:
        # Конвертируем PIL Image в массив numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем пороговую обработку
        _, binary = cv2.threshold(gray, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Удаляем шум
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Увеличиваем контраст
        kernel = np.ones((2,2), np.uint8)
        enhanced = cv2.dilate(denoised, kernel, iterations=1)
        
        return enhanced
        
    except Exception as e:
        raise Exception(f"Ошибка при обработке изображения: {str(e)}")

def recognize_text(image):
    """Распознавание текста на изображении"""
    try:
        # Обработка изображения
        processed_image = process_image(image)
        
        # Настройка параметров распознавания
        custom_config = r'--oem 3 --psm 6'
        
        # Распознавание текста
        text = pytesseract.image_to_string(processed_image, 
                                         config=custom_config)
        
        # Очистка и форматирование результата
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Исправление типичных ошибок распознавания
                line = line.replace('≥', '>=')
                line = line.replace('≤', '<=')
                lines.append(line)
        
        return '\n'.join(lines)
        
    except Exception as e:
        raise Exception(f"Ошибка при распознавании текста: {str(e)}")

def enhance_image_quality(image):
    """Улучшение качества изображения"""
    try:
        # Увеличение размера
        height, width = image.shape[:2]
        scaled = cv2.resize(image, (width*2, height*2), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Улучшение резкости
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(scaled, -1, kernel)
        
        return sharpened
        
    except Exception as e:
        raise Exception(f"Ошибка при улучшении качества: {str(e)}")
