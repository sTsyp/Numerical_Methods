import numpy as np
import matplotlib.pyplot as plt
from funcs import euler_method, runge_kutta_method, find_euler_step_for_rk_error, analytical_solution

# Исходные данные
def f(t, y):
    return 4*t*y-4*t**3

t0 = 0.0
T = 1.0
y0 = -0.5
h = 0.1

t_euler, y_euler = euler_method(f, t0, y0, T, h)

t_rk, y_rk = runge_kutta_method(f, t0, y0, T, h)


t_analytical = np.arange(t0, T + h, h)
y_analytical = analytical_solution(t_analytical)

import pandas as pd

# Построение таблицы значений
data = {
    't': t_analytical,
    'Euler': y_euler,
    'Runge-Kutta': y_rk,
    'Analytical': y_analytical
}
df = pd.DataFrame(data)
print(df)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t_analytical, y_analytical, label='Analytical Solution', color='black')
plt.plot(t_euler, y_euler, label='Euler Method', linestyle='--', color='blue')
plt.plot(t_rk, y_rk, label='Runge-Kutta Method', linestyle='-.', color='red')
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of Numerical and Analytical Solutions')
plt.grid(True)
plt.savefig('plots/7127', dpi=300)


error_euler = np.max(np.abs(y_analytical - y_euler))
error_rk = np.max(np.abs(y_analytical - y_rk))

print(f"Max error (Euler): {error_euler}")
print(f"Max error (Runge-Kutta): {error_rk}")

# Для метода Эйлера с шагом h/2
_, y_euler_half_h = euler_method(f, t0, y0, T, h/2)

# Расчет по правилу Рунге
error_euler_runge = np.max(np.abs(y_euler_half_h[::2] - y_euler))
print(f"Runge error (Euler): {error_euler_runge}")


target_error = error_rk
h_star = find_euler_step_for_rk_error(f, t0, y0, T, target_error)
print(f"Step size h* for Euler to match RK error: {h_star}")