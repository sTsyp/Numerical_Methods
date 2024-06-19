import numpy as np
from optimize import golden_section_minimization
def f(t):
    return np.cos(t**2)

# Заданные параметры
x1 = 0
x2 = 2
epsilon = 1e-6

# Поиск минимума
min_x, min_iterations = golden_section_minimization(f, x1, x2, epsilon)
print("Минимум:", min_x)
print("Число итераций для минимума:", min_iterations)

# Поиск максимума
max_x, max_iterations = golden_section_minimization(lambda x: -f(x), x1, x2, epsilon)
print("Максимум:", max_x)
print("Число итераций для максимума:", max_iterations)
