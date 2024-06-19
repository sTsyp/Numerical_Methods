import numpy as np

def euler_method(f, t0, y0, T, h):
    t_values = np.arange(t0, T + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])

    return t_values, y_values

def runge_kutta_method(f, t0, y0, T, h):
    t_values = np.arange(t0, T + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        k1 = h * f(t_values[i-1], y_values[i-1])
        k2 = h * f(t_values[i-1] + h/2, y_values[i-1] + k1/2)
        k3 = h * f(t_values[i-1] + h/2, y_values[i-1] + k2/2)
        k4 = h * f(t_values[i-1] + h, y_values[i-1] + k3)
        y_values[i] = y_values[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values

def analytical_solution(t):
    return t ** 2 + 0.5 - np.exp(2*t**2)

def find_euler_step_for_rk_error(f, t0, y0, T, target_error):
    h = 0.1
    while True:
        t, y = euler_method(f, t0, y0, T, h)
        y_exact = analytical_solution(t)
        error = np.max(np.abs(y_exact - y))
        if error <= target_error:
            return h
        h /= 2
#########################
def runge_kutta_method2(f, t0, y0, T, h, params):
    t_values = np.arange(t0, T + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        k1 = h * f(t_values[i-1], y_values[i-1], params)
        k2 = h * f(t_values[i-1] + h/2, y_values[i-1] + k1/2, params)
        k3 = h * f(t_values[i-1] + h/2, y_values[i-1] + k2/2, params)
        k4 = h * f(t_values[i-1] + h, y_values[i-1] + k3, params)
        y_values[i] = y_values[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values
def system_ode(t, y, params):
    H, k, m, f = params['H'], params['k'], params['m'], params['f']
    x1, x2 = y
    dx1dt = x2
    dx2dt = (f(t) - H * x2 - k * x1) / m
    return np.array([dx1dt, dx2dt])
#########################
def exact_solution(t):
    return 1 - np.exp(-t) + np.exp(-20 * t)

def find_step_for_accuracy_euler(f, t0, y0, T, exact_solution, epsilon):
    h = 0.15
    while True:
        t, y = euler_method(f, t0, y0, T, h)
        y_exact = exact_solution(t)
        error = np.max(np.abs(y_exact - y))
        if error <= epsilon:
            return h
        h /= 2

def find_step_for_accuracy_rk(f, t0, y0, T, exact_solution, epsilon):
    h = 0.15
    while True:
        t, y = runge_kutta_method(f, t0, y0, T, h)
        y_exact = exact_solution(t)
        error = np.max(np.abs(y_exact - y))
        if error <= epsilon:
            return h
        h /= 2

