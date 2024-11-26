import tkinter as tk
from tkinter import ttk
from tab1_standard import StandardInequalitiesTab
from tab2_uv import UVSystemTab
import sys
import os

class InequalitySystemSolver:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Решатель систем неравенств")
        self.root.geometry("800x600")
        
        # Настройка стиля
        style = ttk.Style()
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TFrame', background='#ffffff')
        
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Инициализация вкладок
        self.tab1 = StandardInequalitiesTab(self.notebook)
        self.tab2 = UVSystemTab(self.notebook)
        
        # Добавление вкладок в notebook
        self.notebook.add(self.tab1, text='Стандартные неравенства')
        self.notebook.add(self.tab2, text='Система U,V')
        
        # Настройка меню
        self.setup_menu()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню файла
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню справки
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)
        
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("О программе")
        about_window.geometry("300x200")
        
        text = """Решатель систем неравенств
Версия 1.0
        
Программа для решения систем неравенств
и построения графиков областей решений."""
        
        label = ttk.Label(about_window, text=text, justify=tk.CENTER)
        label.pack(expand=True)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = InequalitySystemSolver()
    app.run()
