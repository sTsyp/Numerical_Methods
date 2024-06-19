import numpy as np
import matplotlib.pyplot as plt
from funcs import euler_method, runge_kutta_method, find_step_for_accuracy_euler, exact_solution, find_step_for_accuracy_rk

def f(t, y):
    return -20 * y + 20 - 19 * np.exp(-t)

t0 = 0.0
T = 1.5
y0 = 1.0
h = 0.15


t_euler, y_euler = euler_method(f, t0, y0, T, h)

t_rk, y_rk = runge_kutta_method(f, t0, y0, T, h)

t_exact = np.arange(t0, T + h, h)
y_exact = exact_solution(t_exact)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t_exact, y_exact, label='Exact Solution', color='black')
plt.plot(t_euler, y_euler, label='Euler Method', linestyle='--', color='blue')
plt.plot(t_rk, y_rk, label='Runge-Kutta Method', linestyle='-.', color='red')
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of Numerical and Exact Solutions')
plt.grid(True)
plt.savefig('plots/754', dpi=300)

epsilon = 1e-3



h_euler = find_step_for_accuracy_euler(f, t0, y0, T, exact_solution, epsilon)
print(f"Step size for Euler method to achieve accuracy {epsilon}: {h_euler}")


h_rk = find_step_for_accuracy_rk(f, t0, y0, T, exact_solution, epsilon)
print(f"Step size for Runge-Kutta method to achieve accuracy {epsilon}: {h_rk}")