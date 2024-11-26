import re
from sympy import symbols, solve, Eq, parse_expr

def transform_uv_system(system):
    """Преобразует систему из u,v формата в x,y формат"""
    try:
        # Получаем выражения для u и v
        u_expr = system['u']
        v_expr = system['v']
        inequalities = system['inequalities'].strip().split('\n')
        
        if not u_expr or not v_expr or not inequalities:
            raise ValueError("Все поля должны быть заполнены")
        
        # Создаем символьные переменные
        x, y, u, v = symbols('x y u v')
        
        # Парсим выражения u и v
        u_equation = Eq(u, parse_expr(u_expr))
        v_equation = Eq(v, parse_expr(v_expr))
        
        # Решаем систему относительно x и y
        solution = solve((u_equation, v_equation), (x, y))
        
        if not solution:
            raise ValueError("Не удалось выразить x и y через u и v")
        
        # Заменяем u и v на полученные выражения в неравенствах
        transformed_inequalities = []
        for ineq in inequalities:
            if not ineq.strip():
                continue
                
            # Парсим неравенство
            ineq = ineq.strip()
            match = re.match(r'(.*?)(>=|<=|>|<)(.*)', ineq)
            if not match:
                raise ValueError(f"Некорректное неравенство: {ineq}")
            
            left, sign, right = match.groups()
            
            # Заменяем u и v на выражения через x и y
            left_expr = parse_expr(left)
            left_expr = left_expr.subs({u: parse_expr(u_expr), 
                                      v: parse_expr(v_expr)})
            
            # Формируем новое неравенство
            transformed = f"{left_expr} {sign} {right}"
            transformed_inequalities.append(transformed)
        
        return '\n'.join(transformed_inequalities)
        
    except Exception as e:
        raise Exception(f"Ошибка при преобразовании системы: {str(e)}")

def parse_linear_expression(expr):
    """Парсит линейное выражение вида ax+by+c"""
    try:
        # Очищаем выражение от пробелов
        expr = expr.replace(' ', '')
        
        # Заменяем - на +(-) для упрощения парсинга
        expr = expr.replace('-', '+-')
        if expr.startswith('+'):
            expr = expr[1:]
            
        terms = expr.split('+')
        
        coeff_x = 0
        coeff_y = 0
        const = 0
        
        for term in terms:
            if not term:
                continue
                
            if 'x' in term:
                if term == 'x':
                    coeff_x = 1
                elif term == '-x':
                    coeff_x = -1
                else:
                    coeff_x = float(term.replace('x', ''))
            elif 'y' in term:
                if term == 'y':
                    coeff_y = 1
                elif term == '-y':
                    coeff_y = -1
                else:
                    coeff_y = float(term.replace('y', ''))
            else:
                const = float(term)
        
        return coeff_x, coeff_y, const
        
    except Exception as e:
        raise ValueError(f"Ошибка при разборе выражения '{expr}': {str(e)}")
