import numpy as np
from typing import Callable

def deriv(func: Callable[[float], float],
          point: float,
          eps: float = 1e-5) -> float:
    return (func(point + eps) - func(point - eps)) / (2 * eps)


def deriv2(func: Callable[[float], float],
           point: float,
           eps: float = 1e-5) -> float:
    return (func(point + eps) - 2 * func(point) + func(point - eps)) / (eps ** 2)


def grad(f: Callable[[np.array], np.array],
         x: np.array,
         eps: float = 1e-5) -> np.array:
    dim = len(x)
    grad_vector = np.zeros((dim, ), dtype=np.double)
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] += eps
        grad_vector[i] = (f(x + delta) - f(x - delta)) / (eps * 2)
    return grad_vector


def hessian(f: Callable[[np.array], np.array],
            x: np.array,
            eps: float = 1e-5) -> np.array:
    dim = len(x)
    hess = np.zeros((dim, dim), dtype=np.double)
    for i in range(dim):
        i_d = np.zeros(dim)
        i_d[i] += eps
        for j in range(dim):
            j_d = np.zeros(dim)
            j_d[j] += eps
            hess[i, j] = (f(x - i_d - j_d) - f(x + i_d - j_d)
                          - f(x - i_d + j_d) + f(x + i_d + j_d)
                          ) / (4 * eps ** 2)
    return hess


def point_in_area(point, bbox):
    return bbox[0][0] < point[0] < bbox[1][0]\
           and bbox[0][1] < point[1] < bbox[1][1]


def newton_optimize_vec(func: Callable[[np.array], np.array],
                        bbox: tuple[np.array, np.array],
                        start: np.array,
                        eps: float = 1e-5,
                        minimize: bool = True) -> tuple[np.array, int]:
    x = start.astype(np.double)
    point_grad = grad(func, x)
    iter_cnt = 0
    while point_in_area(x, bbox) and np.linalg.norm(point_grad) > eps:
        step = np.linalg.inv(hessian(func, x)).dot(point_grad)
        x -= step if minimize else -step
        point_grad = grad(func, x)
        iter_cnt += 1
    x[0] = max(min(bbox[1][0], x[0]), bbox[0][0])
    x[1] = max(min(bbox[1][1], x[1]), bbox[0][1])
    return x, iter_cnt


def conjucate_grad_minimize(func: Callable[[np.array], np.array],
                            start: np.array,
                            eps: float = 1e-5) -> tuple[np.array, int]:
    x = start.astype(np.double)
    x_prev, h_prev = None, None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None and x_prev is not None:
            denominator = np.linalg.norm(grad(func, x_prev))
            beta = (np.linalg.norm(grad(func, x)) / denominator) ** 2
            h += beta * h_prev
        alpha, _ = newton_optimize_scal(lambda a: func(x + a * h),
                                        interval=(-5, 5),
                                        start=0,
                                        eps=eps * 1e-2)
        x_prev = x.copy()
        x += alpha * h
        h_prev = h.copy()
        iter_cnt += 1

    return x, iter_cnt


def conjucate_grad_quadratic_minimize(func: Callable[[np.array], np.array],
                                      func_matrix: np.array,
                                      start: np.array,
                                      eps: float = 1e-5) -> tuple[np.array, int]:
    x = start.astype(np.double)
    h_prev = None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None:
            denominator = ((func_matrix @ h_prev) @ h_prev)
            beta = ((func_matrix @ h_prev) @ grad(func, x)) / denominator
            h += beta * h_prev
        alpha, _ = newton_optimize_scal(lambda a: func(x + a * h),
                                        interval=(-10, 10),
                                        start=0,
                                        eps=eps * 1e-2)
        x += alpha * h
        h_prev = h.copy()
        iter_cnt += 1

    return x, iter_cnt


def newton_optimize_scal(func: Callable,
                         interval: tuple[float, float],
                         start: float,
                         eps: float = 1e-6,
                         minimize: bool = True,
                         max_iter: int = 1000) -> tuple[float, int]:
    (a, b), x = interval, start
    cnt = 0
    while cnt < max_iter and a < x < b and abs(deriv(func, x)) > eps:
        step = deriv(func, x) / deriv2(func, x)
        x += -step if minimize else step
        cnt += 1
    x = min(b, max(a, x))
    return x, cnt


def golden_section_minimization(f, a, b, epsilon):
    golden_ratio = (np.sqrt(5) - 1) / 2
    x1 = a + (1 - golden_ratio) * (b - a)
    x2 = a + golden_ratio * (b - a)
    iterations = 0
    while abs(b - a) > epsilon:
        if f(x1) < f(x2):
            b = x2
            x2 = x1
            x1 = a + (1 - golden_ratio) * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + golden_ratio * (b - a)
        iterations += 1
    return (a + b) / 2, iterations

    
