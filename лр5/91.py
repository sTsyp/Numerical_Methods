import numpy as np
import matplotlib.pyplot as plt

from optimize import newton_optimize_scal


def f(t):
    return 2 * np.sin(t) - 3*np.cos(t)


interval = (-1, 1)

minima, iter_min = newton_optimize_scal(func=f,
                                        interval=interval,
                                        start=0.0,
                                        eps=1e-6,
                                        minimize=True)
maxima, iter_max = newton_optimize_scal(func=f,
                                        interval=interval,
                                        start=-1.0,
                                        eps=1e-6,
                                        minimize=False)
right_maxima, right_iter_max = newton_optimize_scal(func=f,
                                                    interval=interval,
                                                    start=0.75,
                                                    eps=1e-6,
                                                    minimize=False)
t = np.linspace(start=interval[0], stop=interval[1], num=100)
y = f(t)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(t, y, label='$f(x)$')
plt.scatter(minima, f(minima),
            s=70,
            color='red',
            label=f'$min$ $f(x)$, {iter_min} iterations')
plt.scatter(maxima, f(maxima),
            s=70,
            color='green',
            label=f'$max$ $f(x)$, {iter_max} iterations')
plt.scatter(right_maxima, f(right_maxima),
            s=70,
            color='green',
            label=f'$max$ $f(x)$, {right_iter_max} iterations')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('plots/newton.png', dpi=300)
