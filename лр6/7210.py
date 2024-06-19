import numpy as np
import matplotlib.pyplot as plt
from funcs import runge_kutta_method2, system_ode

# Функция f(t)
def f1(t):
    return -np.sqrt(t)

def f2(t):
    return t

def f3(t):
    return np.sqrt(t)

# Параметры для каждого набора
params = [
    {'H': 0.5, 'k': 1, 'm': 0.5, 'f': f1, 'x0': 0, 'v0': -10, 'T': 15},
    {'H': 0.5, 'k': 1, 'm': 0.5, 'f': f2, 'x0': 0, 'v0': 10, 'T': 15},
    {'H': 2, 'k': 5, 'm': 0.5, 'f': f3, 'x0': 0, 'v0': 10, 'T': 15}
]


# Решение для каждого набора параметров
results = []
for param in params:
    y0 = [param['x0'], param['v0']]
    t_values, y_values = runge_kutta_method2(system_ode, 0, y0, param['T'], 0.1, param)
    results.append((t_values, y_values))

# Построение графиков
plt.figure(figsize=(15, 10))
for i, (t_values, y_values) in enumerate(results):
    plt.subplot(3, 1, i+1)
    plt.plot(t_values, y_values[:, 0], label=f'Set {i+1}')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(f'Set {i+1} Solution')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('plots/7210', dpi=300)

# Определение максимальных и минимальных значений
for i, (t_values, y_values) in enumerate(results):
    x_values = y_values[:, 0]
    max_x = np.max(x_values)
    min_x = np.min(x_values)
    t_max = t_values[np.argmax(x_values)]
    t_min = t_values[np.argmin(x_values)]

    print(f"Set {i+1}:")
    print(f"Max x(t): {max_x} at t = {t_max}")
    print(f"Min x(t): {min_x} at t = {t_min}")
    print()