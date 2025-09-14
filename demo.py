from paddleocr import PaddleOCR
import gradio as gr
import json
import ollama
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os
import fitz  # PyMuPDF для работы с PDF
from pdf2image import convert_from_path  # для конвертации PDF в изображения
import tempfile
import shutil
import re
import pytesseract
from pytesseract import Output

# Указываем путь к poppler
POPPLER_PATH = r"D:\ustonovlenie\progi\poppler-25.07.0\Library\bin"
os.environ["PATH"] += os.pathsep + POPPLER_PATH

# Инициализируем модель PaddleOCR для нескольких языков
ocr = PaddleOCR(use_angle_cls=True, lang='en')
ocr_ru = PaddleOCR(use_angle_cls=True, lang='ru')
ocr_kz = PaddleOCR(use_angle_cls=True, lang='korean')

def is_scanned_pdf(pdf_path):
    """Проверяет, является ли PDF сканом (содержит только изображения)"""
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page_num in range(min(3, len(doc))):  # Проверяем первые 3 страницы
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            text_content += text
            
            # Проверяем, есть ли на странице изображения
            image_list = page.get_images()
            if image_list:
                # Если есть изображения и мало текста, вероятно это скан
                if len(text) < 50 and len(image_list) > 0:
                    return True
        
        # Если текст содержит много нестандартных символов или мало текста
        if len(text_content) < 100:
            return True
            
        # Проверяем структуру текста - в сканированных PDF текст часто имеет странную структуру
        lines = text_content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
        if avg_line_length < 10:  # Очень короткие строки могут указывать на сканированный текст
            return True
            
        return False  # Если есть достаточно текста с нормальной структурой
        
    except Exception as e:
        print(f"Ошибка при проверке типа PDF: {e}")
        return True  # В случае ошибки считаем сканированным

def pdf_to_images(pdf_path):
    """Конвертирует PDF в изображения с улучшенными настройками"""
    try:
        images = convert_from_path(
            pdf_path, 
            poppler_path=POPPLER_PATH, 
            dpi=400,  # Увеличиваем DPI для лучшего качества
            thread_count=4
        )
        image_paths = []
        
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image_path = tmp_file.name
                image.save(image_path, 'PNG', dpi=(400, 400))
                image_paths.append(image_path)
            
        return image_paths
    except Exception as e:
        print(f"Ошибка конвертации PDF: {e}")
        try:
            images = convert_from_path(pdf_path, dpi=400)
            image_paths = []
            
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image_path = tmp_file.name
                    image.save(image_path, 'PNG', dpi=(400, 400))
                    image_paths.append(image_path)
                    
            return image_paths
        except Exception as e2:
            print(f"Ошибка конвертации PDF (без пути): {e2}")
            return None

def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF напрямую (для текстовых PDF)"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                full_text += f"--- Страница {page_num + 1} ---\n{text}\n\n"
            
        return full_text if full_text.strip() else "Текст не найден в PDF"
    except Exception as e:
        return f"Ошибка извлечения текста из PDF: {str(e)}"

def enhanced_preprocess_image(image_path):
    """Улучшенная предобработка изображения для OCR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Конвертируем в оттенки серого
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Увеличиваем разрешение
        height, width = gray.shape
        if width < 2000 or height < 2000:
            gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # Пробуем разные методы бинаризации
        methods = []
        
        # Метод 1: Адаптивная бинаризация
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        methods.append(binary1)
        
        # Метод 2: Otsu's binarization
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, binary2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(binary2)
        
        # Метод 3: Простой threshold
        _, binary3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        methods.append(binary3)
        
        # Сохраняем все варианты для тестирования
        processed_paths = []
        for i, processed_img in enumerate(methods):
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, processed_img)
                processed_paths.append(temp_path)
        
        return processed_paths
        
    except Exception as e:
        print(f"Ошибка улучшенной предобработки: {e}")
        return [image_path]

def extract_text_with_tesseract_enhanced(image_path, languages=['rus+eng+kaz']):
    """Улучшенный Tesseract с поддержкой нескольких языков"""
    try:
        best_text = ""
        
        # Пробуем разные комбинации языков
        language_combinations = [
            'rus+eng+kaz',
            'rus+eng',
            'kaz+rus',
            'eng',
            'rus',
            'kaz'
        ]
        
        # Пробуем разные PSM режимы
        psm_modes = ['6', '3', '4', '8', '13']
        
        for lang_combo in language_combinations:
            for psm in psm_modes:
                try:
                    config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(
                        Image.open(image_path), 
                        lang=lang_combo, 
                        config=config
                    )
                    
                    if text and len(text.strip()) > len(best_text.strip()):
                        best_text = text.strip()
                        print(f"Найден лучший текст: {lang_combo}, PSM {psm}, символов: {len(best_text)}")
                        
                except Exception as e:
                    continue
        
        return best_text if best_text else ""
        
    except Exception as e:
        print(f"Ошибка улучшенного Tesseract: {e}")
        return ""

def extract_text_from_scanned_pdf(pdf_path):
    """Извлекает текст из сканированного PDF с улучшенным OCR"""
    try:
        full_text = ""
        image_paths = pdf_to_images(pdf_path)
        
        if not image_paths:
            return "Ошибка конвертации PDF в изображения"
        
        for i, img_path in enumerate(image_paths):
            print(f"Обработка страницы {i+1}...")
            
            # Улучшенная предобработка - получаем несколько вариантов
            processed_paths = enhanced_preprocess_image(img_path)
            
            best_page_text = ""
            
            # Тестируем все обработанные варианты изображения
            for processed_path in processed_paths:
                try:
                    # Сначала пробуем Tesseract с улучшенными настройками
                    text = extract_text_with_tesseract_enhanced(processed_path)
                    
                    if text and len(text) > len(best_page_text):
                        best_page_text = text
                        
                    # Также пробуем PaddleOCR как fallback
                    paddle_text = extract_text_with_paddleocr(processed_path, 'ru')
                    if paddle_text and len(paddle_text) > len(best_page_text):
                        best_page_text = paddle_text
                        
                except Exception as e:
                    print(f"Ошибка обработки изображения: {e}")
                    continue
                
                # Очистка временного файла обработки
                if os.path.exists(processed_path) and processed_path != img_path:
                    os.remove(processed_path)
            
            if best_page_text:
                full_text += f"--- Страница {i+1} ---\n{best_page_text}\n\n"
                print(f"Страница {i+1}: распознано {len(best_page_text)} символов")
            else:
                full_text += f"--- Страница {i+1} ---\nТекст не распознан\n\n"
            
            # Очистка исходного изображения
            if os.path.exists(img_path):
                os.remove(img_path)
                
        return full_text if full_text.strip() else "Текст не распознан в сканированном PDF"
        
    except Exception as e:
        return f"Ошибка обработки сканированного PDF: {str(e)}"

def extract_text_with_paddleocr(image_path, language='ru'):
    """Извлекает текст с помощью PaddleOCR"""
    try:
        if language == 'ru':
            ocr_model = ocr_ru
        elif language == 'en':
            ocr_model = ocr
        elif language == 'kz':
            ocr_model = ocr_kz
        else:
            ocr_model = ocr_ru
        
        result = ocr_model.ocr(image_path)
        full_text = ""
        
        if result:
            for page in result:
                if page:
                    for line in page:
                        if line and len(line) > 1:
                            text_info = line[1]
                            if text_info and len(text_info) > 0:
                                text = text_info[0]
                                full_text += f"{text}\n"
        
        return full_text.strip()
        
    except Exception as e:
        print(f"Ошибка PaddleOCR: {e}")
        return ""

def process_with_llm(raw_text):
    """Использует локальную LLM через Ollama для структурирования текста"""
    try:
        # Увеличиваем лимит текста для LLM
        text_for_llm = raw_text[:15000]  # Увеличили лимит
        
        prompt = f"""
        Проанализируй текст договора и извлеки ключевую информацию в формате JSON.
        Поля: contract_number, contract_date, seller, buyer, total_amount, currency, 
        delivery_terms, payment_terms, quality_terms, claims_period.
        
        Если какая-то информация отсутствует, укажи "не указано".
        
        Текст договора: {text_for_llm}
        """

        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": prompt}],
            options={'temperature': 0.1}
        )

        return response['message']['content'].strip()

    except Exception as e:
        return f"Ошибка LLM: {str(e)}"

def gradio_interface(file):
    """Функция для Gradio интерфейса"""
    if file is None:
        return "Пожалуйста, загрузите файл"
    
    file_path = file.name
    
    try:
        if file_path.lower().endswith('.pdf'):
            is_scanned = is_scanned_pdf(file_path)
            
            if is_scanned:
                print("Обнаружен сканированный PDF - используем улучшенный OCR...")
                raw_text = extract_text_from_scanned_pdf(file_path)
                pdf_type = "сканированный PDF"
            else:
                print("Обнаружен текстовый PDF - извлекаем текст напрямую...")
                raw_text = extract_text_from_pdf(file_path)
                pdf_type = "текстовый PDF"
                
        else:
            # Для изображений используем первый вариант предобработки
            processed_paths = enhanced_preprocess_image(file_path)
            raw_text = ""
            
            for processed_path in processed_paths:
                text = extract_text_with_tesseract_enhanced(processed_path)
                if text and len(text) > len(raw_text):
                    raw_text = text
                
                if os.path.exists(processed_path) and processed_path != file_path:
                    os.remove(processed_path)
            
            pdf_type = "изображение"
        
        # Сохраняем полный текст в файл для отладки
        with open("full_recognized_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
        
        # Обработка с LLM
        structured_data = process_with_llm(raw_text)
        
        # Показываем полный текст в интерфейсе
        result = f"ТИП ФАЙЛА: {pdf_type}\n\n"
        result += f"ПОЛНЫЙ РАСПОЗНАННЫЙ ТЕКСТ:\n{raw_text}\n\n"
        result += f"АНАЛИЗ ДОГОВОРА:\n{structured_data}"
        
        return result
        
    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}"

# Обновляем интерфейс для поддержки большего количества текста
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Загрузите PDF или изображение документа"),
    outputs=gr.Textbox(label="Результат обработки", lines=50),
    title="OCR для банковских документов и договоров",
    description="Загрузите PDF или изображение документа для распознавания (поддержка рус/англ/каз языков)"
)

if __name__ == "__main__":
    print("Запуск приложения...")
    print(f"Используемый путь к poppler: {POPPLER_PATH}")
    iface.launch(share=True)