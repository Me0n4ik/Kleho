import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def plot_inequalities(inequalities_text, frame):
    """Построение графика системы неравенств"""
    # Очищаем фрейм
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Создаем фигуру и оси
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    # Настройка осей
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('График системы неравенств')
    
    # Создаем сетку точек
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    
    # Парсим и строим каждое неравенство
    inequalities = inequalities_text.strip().split('\n')
    combined_mask = None
    
    for inequality in inequalities:
        if not inequality.strip():
            continue
            
        mask = parse_inequality(inequality, X, Y)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask & mask
    
    # Отображаем область решений
    if combined_mask is not None:
        ax.imshow(combined_mask, extent=[-10, 10, -10, 10], 
                 origin='lower', cmap='Blues', alpha=0.3)
    
    # Добавляем оси координат
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Создаем канвас и размещаем его
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def parse_inequality(inequality, X, Y):
    """Парсинг отдельного неравенства и создание маски"""
    # Очищаем неравенство от пробелов
    inequality = inequality.replace(' ', '')
    
    # Определяем знак неравенства
    if '>=' in inequality:
        sign = '>='
        left, right = inequality.split('>=')
    elif '<=' in inequality:
        sign = '<='
        left, right = inequality.split('<=')
    elif '>' in inequality:
        sign = '>'
        left, right = inequality.split('>')
    elif '<' in inequality:
        sign = '<'
        left, right = inequality.split('<')
    else:
        raise ValueError(f"Неподдерживаемый знак неравенства в: {inequality}")
    
    # Вычисляем значение левой части
    Z = eval_expression(left, X, Y)
    
    # Вычисляем значение правой части
    right_val = float(right)
    
    # Создаем маску в зависимости от знака
    if sign in ['>=', '>']:
        return Z >= right_val
    else:
        return Z <= right_val

def eval_expression(expr, X, Y):
    """Вычисление значения выражения для каждой точки сетки"""
    # Заменяем переменные на массивы numpy
    expr = expr.replace('x', 'X').replace('y', 'Y')
    return eval(expr)
