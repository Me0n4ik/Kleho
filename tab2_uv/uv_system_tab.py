import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import pytesseract
from utils.plot_helper import plot_inequalities
from utils.image_processor import process_image, recognize_text

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import cv2

class UVSystemTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_image = None
        self.image_path = None
        # Инициализация переменных
        self.current_image = None
        self.fig1 = None
        self.fig2 = None
        self.ax1 = None
        self.ax2 = None
        self.canvas1 = None
        self.canvas2 = None
        
        # Установка пути к Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.setup_gui()
        
    def setup_gui(self):
        # Главный контейнер с тремя колонками
        main_container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель для ввода неравенств
        left_frame = ttk.Frame(main_container, width=400)
        main_container.add(left_frame, weight=1)
        
        # Центральная панель для первого графика
        self.center_frame = ttk.Frame(main_container, width=500)
        main_container.add(self.center_frame, weight=2)
        
        # Правая панель для второго графика
        self.right_frame = ttk.Frame(main_container, width=500)
        main_container.add(self.right_frame, weight=2)
        
        # Настройка левой панели
        self.setup_left_panel(left_frame)
        
        # Инициализация графиков
        self.init_plots()
        
    def transform_uv_to_xy(self, text):
        """
        Преобразует систему из u,v координат в x,y координаты
        """
        try:
            # Разбиваем текст на строки и удаляем пустые строки
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if len(lines) < 3:  # Минимум 2 уравнения и 1 неравенство
                raise ValueError("Недостаточно строк для преобразования")

            def parse_equation(eq):
                """Парсит уравнение вида u=ax+by или v=ax+by"""
                # Очищаем от пробелов и разделяем на левую и правую части
                eq = eq.replace(' ', '')
                left, right = eq.split('=')
                
                # Начальные коэффициенты
                x_coef = y_coef = const = 0
                
                # Добавляем '+' если выражение не начинается с '+' или '-'
                if not right.startswith(('+', '-')):
                    right = '+' + right
                    
                # Разбиваем на члены
                terms = right.replace('-', '+-').split('+')
                
                for term in terms:
                    if not term:
                        continue
                    if 'x' in term:
                        term = term.replace('x', '')
                        x_coef = 1 if term in ['', '+'] else (-1 if term == '-' else float(term))
                    elif 'y' in term:
                        term = term.replace('y', '')
                        y_coef = 1 if term in ['', '+'] else (-1 if term == '-' else float(term))
                    else:
                        try:
                            const = float(term)
                        except:
                            continue
                            
                return x_coef, y_coef, const

            # Получаем коэффициенты из первых двух уравнений
            u_x, u_y, u_c = parse_equation(lines[0])  # u = ax + by + c
            v_x, v_y, v_c = parse_equation(lines[1])  # v = dx + ey + f
            
            def parse_inequality(ineq):
                """Парсит неравенство вида au+bv<c или au+bv>c"""
                # Определяем знак неравенства
                if '<=' in ineq:
                    left, right = ineq.split('<=')
                    sign = '<='
                elif '>=' in ineq:
                    left, right = ineq.split('>=')
                    sign = '>='
                elif '<' in ineq:
                    left, right = ineq.split('<')
                    sign = '<'
                elif '>' in ineq:
                    left, right = ineq.split('>')
                    sign = '>'
                else:
                    raise ValueError(f"Неподдерживаемый знак неравенства в {ineq}")

                # Парсим коэффициенты перед u и v
                left = left.replace(' ', '')
                if not left.startswith(('+', '-')):
                    left = '+' + left
                terms = left.replace('-', '+-').split('+')
                
                u_coef = v_coef = 0
                for term in terms:
                    if not term:
                        continue
                    if 'u' in term:
                        term = term.replace('u', '')
                        u_coef = 1 if term in ['', '+'] else (-1 if term == '-' else float(term))
                    elif 'v' in term:
                        term = term.replace('v', '')
                        v_coef = 1 if term in ['', '+'] else (-1 if term == '-' else float(term))
                
                return u_coef, v_coef, float(right), sign

            # Преобразуем каждое неравенство
            transformed_inequalities = []
            for ineq in lines[2:]:
                u_coef, v_coef, right, sign = parse_inequality(ineq)
                
                # Подставляем выражения для u и v
                # (au + bv < c) -> (a(u_x*x + u_y*y + u_c) + b(v_x*x + v_y*y + v_c) < c)
                x_coef = u_coef * u_x + v_coef * v_x
                y_coef = u_coef * u_y + v_coef * v_y
                const = u_coef * u_c + v_coef * v_c - right
                
                # Формируем новое неравенство
                terms = []
                if x_coef != 0:
                    terms.append(f"{x_coef:+g}x".replace("+", ""))
                if y_coef != 0:
                    terms.append(f"{y_coef:+g}y")
                if const != 0:
                    terms.append(f"{const:+g}")
                    
                left_side = "".join(terms)
                if left_side.startswith("+"):
                    left_side = left_side[1:]
                    
                transformed_inequalities.append(f"{left_side} {sign} 0")

            return "\n".join(transformed_inequalities)
            
        except Exception as e:
            print(f"Ошибка при преобразовании системы: {str(e)}")
            return None

    def transform_coordinates(self):
        try:
            # Получаем текст из поля ввода
            text = self.inequalities_text.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("Предупреждение", "Введите систему неравенств")
                return
                
            # Преобразуем систему
            transformed_text = self.transform_uv_to_xy(text)
            
            if transformed_text:
                # Очищаем текстовое поле и вставляем преобразованный текст
                self.inequalities_text.delete(1.0, tk.END)
                self.inequalities_text.insert(tk.END, transformed_text)
                self.status_label.config(text="Система преобразована в x,y координаты")
            else:
                messagebox.showerror("Ошибка", "Не удалось преобразовать систему")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при преобразовании: {str(e)}")

    def setup_left_panel(self, parent):
        # Кнопки управления
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Загрузить изображение", 
                command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Распознать текст", 
                command=self.recognize_text).pack(side=tk.LEFT, padx=2)
        
        # Добавляем новую кнопку для преобразования координат
        ttk.Button(control_frame, text="Преобразовать в x,y", 
            command=self.transform_coordinates).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Построить график", 
                command=self.plot_inequalities).pack(side=tk.LEFT, padx=2)
        # Добавляем новую кнопку
        ttk.Button(control_frame, text="Сгенерировать ответ", 
                command=self.generate_answer).pack(side=tk.LEFT, padx=2)
    

        # Фрейм для изображения
        self.image_frame = ttk.LabelFrame(parent, text="Изображение")
        self.image_frame.pack(fill=tk.X, padx=5, pady=5)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=5, pady=5)
        
        # Текстовое поле
        text_frame = ttk.LabelFrame(parent, text="Система неравенств")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Добавляем скроллбар
        scroll = ttk.Scrollbar(text_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.inequalities_text = tk.Text(text_frame, height=10, yscrollcommand=scroll.set)
        self.inequalities_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.config(command=self.inequalities_text.yview)
        
        # Статус
        self.status_label = ttk.Label(parent, text="Готов к работе")
        self.status_label.pack(pady=5, padx=5)

    def generate_answer(self):
        try:
            # Получаем текущий текст неравенств
            text = self.inequalities_text.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("Предупреждение", "Введите систему неравенств")
                return

            # Получаем все неравенства
            inequalities = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Находим центр закрашенной области
            mask, _ = self.process_inequalities(text)
            if mask is None:
                messagebox.showerror("Ошибка", "Не удалось построить область")
                return
                
            # Находим центр закрашенной области
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                messagebox.showerror("Ошибка", "Область решений пуста")
                return
                
            center_x = np.mean(x_indices) * 20 / len(mask[0]) - 10
            center_y = np.mean(y_indices) * 20 / len(mask) - 10
            
            # Определяем знаки для каждого неравенства
            answer = ""
            
            for ineq in inequalities:
                parsed = self.parse_inequality(ineq)
                if parsed is None:
                    continue
                
                a, b, c, _ = parsed
                value = a * center_x + b * center_y - c
                
                if value > 0:
                    answer += 'g'
                else:
                    answer += 'l'

            # Создаем новое окно
            answer_window = tk.Toplevel(self)
            answer_window.title("Ответ")
            answer_window.geometry("200x100")
            
            # Создаем текстовое поле с ответом
            answer_text = tk.Text(answer_window, height=1, width=len(answer)+2, 
                                font=('Arial', 16))
            answer_text.pack(expand=True, padx=10, pady=10)
            answer_text.insert('1.0', answer)
            
            # Функция для копирования текста
            def copy_answer():
                answer_window.clipboard_clear()
                answer_window.clipboard_append(answer)
                answer_window.update()
                
            # Добавляем кнопку копирования
            copy_button = ttk.Button(answer_window, text="Копировать", 
                                command=copy_answer)
            copy_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации ответа: {str(e)}")


    def init_plots(self):
        # Создаем фигуры и оси для обоих графиков
        self.fig1 = Figure(figsize=(6, 6))
        self.ax1 = self.fig1.add_subplot(111)
        self.setup_plot(self.ax1, "График системы неравенств")
        
        self.fig2 = Figure(figsize=(6, 6))
        self.ax2 = self.fig2.add_subplot(111)
        self.setup_plot(self.ax2, "Интерактивный график")
        
        # Создаем канвасы
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.center_frame)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.right_frame)
        
        # Добавляем тулбары
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.center_frame)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.right_frame)
        
        # Размещаем элементы
        self.toolbar1.pack(side=tk.TOP, fill=tk.X)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar2.pack(side=tk.TOP, fill=tk.X)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем обработчик клика для второго графика
        self.canvas2.mpl_connect('button_press_event', self.on_plot_click)

    def setup_plot(self, ax, title):
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_aspect('equal')

    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                    ("Все файлы", "*.*")
                ]
            )
            
            if not file_path:
                return
                
            image = Image.open(file_path)
            self.current_image = image
            
            # Создаем превью
            display_size = (300, 300)
            preview = image.copy()
            preview.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            self.status_label.config(text="Изображение загружено")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке изображения: {str(e)}")

    def recognize_text(self):
        if not self.current_image:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
                
        try:
# Создаем копию изображения для обработки
            image = self.current_image.copy()
            
            # Конвертируем в RGB если изображение в другом формате
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Увеличиваем размер изображения
            scale_factor = 2
            width, height = image.size
            image = image.resize((width * scale_factor, height * scale_factor), 
                            Image.Resampling.LANCZOS)
            
            # Преобразуем в массив numpy
            img_array = np.array(image)
            
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Применяем адаптивную пороговую обработку
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Удаляем шум
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Увеличиваем толщину текста
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            # Настройки для Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789xy<>=-+.?vu \'\""'
            
            # Распознаем текст
            text = pytesseract.image_to_string(binary, 
                                            lang='eng', 
                                            config=custom_config)

            # Обрабатываем распознанный текст
            lines = []
            for line in text.split('\n'):
                # Пропускаем пустые строки
                if not line.strip():
                    continue
                    
                # Очищаем строку от специальных символов и лишних пробелов
                line = line.strip()
                line = line.replace('"', '')  # Удаляем кавычки
                line = line.replace("'", '')  # Удаляем одинарные кавычки
                line = line.replace('"', '')  # Удаляем двойные кавычки
                line = line.replace('`', '')  # Удаляем обратные кавычки
                line = line.replace('—', '-')  # Заменяем длинное тире
                line = line.replace('–', '-')  # Заменяем среднее тире
                line = line.replace('≤', '<=')  # Исправляем знак меньше или равно
                line = line.replace('≥', '>=')  # Исправляем знак больше или равно
                line = line.replace(' ', '')    # Удаляем пробелы
                
                # Исправляем возможные ошибки распознавания
                line = line.replace('|', 'l')   # Вертикальная черта на l
                line = line.replace('I', '1')   # I на 1
                line = line.replace('O', '0')   # O на 0
                
                if '?' in line:
                    line = line.replace('?', '<')
                    
                # Проверяем наличие знаков неравенства или знака вопроса
                if any(sign in line for sign in ['<', '>', '<=', '>=', '?', '=']):
                    lines.append(line)
            
            # Объединяем обработанные строки
            processed_text = '\n'.join(lines)
            
            # Очищаем текстовое поле и вставляем распознанный текст
            self.inequalities_text.delete(1.0, tk.END)
            self.inequalities_text.insert(tk.END, processed_text)
            
            # Проверяем результат
            if not processed_text.strip():
                raise Exception("Не удалось распознать текст")
                
            self.status_label.config(text="Текст распознан")
                
        except ValueError as ve:
            if "No closing quotation" in str(ve):
                # Если возникла ошибка с кавычками, пытаемся очистить текст от кавычек
                try:
                    processed_text = processed_text.replace('"', '').replace("'", '')
                    self.inequalities_text.delete(1.0, tk.END)
                    self.inequalities_text.insert(tk.END, processed_text)
                    self.status_label.config(text="Текст распознан (удалены кавычки)")
                except:
                    messagebox.showerror("Ошибка", "Не удалось обработать кавычки в тексте")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при распознавании: {str(e)}")
    
    def parse_inequality(self, inequality_str, flip_signs=False):
        """Парсинг строки неравенства и получение коэффициентов"""
        try:
            # Очистка от пробелов
            inequality_str = inequality_str.replace(' ', '')
            
            # Определение знака неравенства
            if '>=' in inequality_str:
                left, right = inequality_str.split('>=')
                sign = '<=' if flip_signs else '>='
            elif '<=' in inequality_str:
                left, right = inequality_str.split('<=')
                sign = '>=' if flip_signs else '<='
            elif '>' in inequality_str:
                left, right = inequality_str.split('>')
                sign = '<' if flip_signs else '>'
            elif '<' in inequality_str:
                left, right = inequality_str.split('<')
                sign = '>' if flip_signs else '<'
            else:
                raise ValueError("Знак неравенства не найден")

            # Обработка левой части
            left = left.replace('-', '+-').replace('++', '+')
            if left.startswith('+'):
                left = left[1:]
            terms = left.split('+')
            
            a, b = 0, 0
            for term in terms:
                if term:
                    if 'x' in term:
                        if term == 'x': a = 1
                        elif term == '-x': a = -1
                        else: a = float(term.replace('x', ''))
                    elif 'y' in term:
                        if term == 'y': b = 1
                        elif term == '-y': b = -1
                        else: b = float(term.replace('y', ''))
            
            # Обработка правой части
            c = float(right)
            
            return a, b, c, sign
            
        except Exception as e:
            print(f"Ошибка при разборе неравенства '{inequality_str}': {str(e)}")
            return None

    def process_inequalities(self, inequalities_text, flip_signs=False):
        """Обработка всех неравенств и создание маски"""
        try:
            inequalities = [line.strip() for line in inequalities_text.split('\n') 
                          if line.strip()]
            
            if not inequalities:
                return None, []
            
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(x, y)
            
            combined_mask = None
            lines_info = []
            
            for ineq in inequalities:
                result = self.parse_inequality(ineq, flip_signs)
                if result is None:
                    continue
                    
                a, b, c, sign = result
                lines_info.append((a, b, c))
                
                # Вычисление маски
                Z = a*X + b*Y - c
                if sign in ['>=', '>']:
                    mask = Z >= 0
                else:
                    mask = Z <= 0
                
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask & mask
                    
            return combined_mask, lines_info
            
        except Exception as e:
            print(f"Ошибка при обработке неравенств: {str(e)}")
            return None, []

    def plot_inequalities(self):
        try:
            # Очищаем оба графика
            self.ax1.clear()
            self.ax2.clear()
            self.setup_plot(self.ax1, "График системы неравенств")
            self.setup_plot(self.ax2, "Интерактивный график")
            
            # Получаем текст неравенств
            text = self.inequalities_text.get(1.0, tk.END)
            
            # Обрабатываем неравенства
            mask, lines = self.process_inequalities(text)
            
            if mask is not None:
                # Отображаем область решений на обоих графиках
                self.ax1.imshow(mask, extent=[-10, 10, -10, 10], 
                              origin='lower', cmap='Blues', alpha=0.3)
                self.ax2.imshow(mask, extent=[-10, 10, -10, 10], 
                              origin='lower', cmap='Blues', alpha=0.3)
                
                # Рисуем линии неравенств на обоих графиках
                x = np.linspace(-10, 10, 100)
                for a, b, c in lines:
                    if b != 0:  # Если это не вертикальная линия
                        y = (-a*x + c)/b
                        self.ax1.plot(x, y, '-', linewidth=1, color='red')
                        self.ax2.plot(x, y, '-', linewidth=1, color='red')
                    elif a != 0:  # Если это вертикальная линия
                        self.ax1.axvline(x=c/a, color='red', linewidth=1)
                        self.ax2.axvline(x=c/a, color='red', linewidth=1)
                
                self.status_label.config(text="Графики построены")
            else:
                self.status_label.config(text="Ошибка при построении графиков")
            
            # Обновляем оба графика
            self.canvas1.draw()
            self.canvas2.draw()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графиков: {str(e)}")

    def generate_alternative_systems(self, text):
        """Генерирует все возможные варианты системы неравенств"""
        inequalities = [line.strip() for line in text.split('\n') 
                       if line.strip()]
        n = len(inequalities)
        alternative_systems = []
        
        # Перебираем все возможные комбинации знаков
        for i in range(2**n):
            new_system = []
            for j, ineq in enumerate(inequalities):
                # Определяем, нужно ли менять знак для данного неравенства
                flip = (i >> j) & 1
                if flip:
                    # Меняем знак неравенства
                    if '>=' in ineq:
                        new_ineq = ineq.replace('>=', '<=')
                    elif '<=' in ineq:
                        new_ineq = ineq.replace('<=', '>=')
                    elif '>' in ineq:
                        new_ineq = ineq.replace('>', '<')
                    elif '<' in ineq:
                        new_ineq = ineq.replace('<', '>')
                else:
                    new_ineq = ineq
                
                new_system.append(new_ineq)
            
            alternative_systems.append('\n'.join(new_system))
        
        return alternative_systems

    def on_plot_click(self, event):
        if event.inaxes != self.ax2:
            return
        
        x, y = event.xdata, event.ydata
        
        # Получаем текущий текст неравенств
        current_text = self.inequalities_text.get(1.0, tk.END)
        
        # Генерируем все возможные варианты систем
        alternative_systems = self.generate_alternative_systems(current_text)
        
        # Проверяем каждую систему
        for system in alternative_systems:
            mask, _ = self.process_inequalities(system)
            if mask is not None:
                i = int((y + 10) * len(mask) / 20)
                j = int((x + 10) * len(mask[0]) / 20)
                if 0 <= i < len(mask) and 0 <= j < len(mask[0]):
                    if mask[i, j]:
                        # Создаем новое окно для отображения системы
                        window = tk.Toplevel(self)
                        window.title("Система неравенств для выбранной области")
                        window.geometry("300x200")
                        
                        text_widget = tk.Text(window, height=10, width=40)
                        text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
                        text_widget.insert(tk.END, system)
                        
                        self.status_label.config(text="Найдена подходящая система неравенств")
                        return
        
        self.status_label.config(text="Для данной точки не найдено подходящей системы")
