from __future__ import annotations
import math
import numpy as np
from constructive_number import ConstructiveNumber, cn_abs, cn_sin, cn_cos, cn_sqrt, cn_round


class Objective:
    name = "Objective"
    dimension = 0
    bounds = None
    optimum = None
    optimum_value = 0.0
    supports_derivatives = False

    def __call__(self, x):
        raise NotImplementedError

    def value(self, x):
        v = self(x)
        return float(v.midpoint if isinstance(v, ConstructiveNumber) else v)

    def cn_value(self, x, eps=1e-8):
        xx = [ConstructiveNumber.from_x_eps(float(v), eps) for v in x]
        return self(xx)

    def gradient(self, x):
        raise NotImplementedError

    def hessian(self, x):
        raise NotImplementedError


class Rastrigin2(Objective):
    name = "Rastrigin-2"
    dimension = 2
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    optimum = np.array([0.0, 0.0])
    optimum_value = 0.0
    supports_derivatives = True

    def __call__(self, x):
        x0, x1 = x
        return 20 + x0*x0 + x1*x1 - 10*(cn_cos(2*math.pi*x0) + cn_cos(2*math.pi*x1))

    def gradient(self, x):
        x0, x1 = x
        return [
            2*x0 + 20*math.pi*cn_sin(2*math.pi*x0),
            2*x1 + 20*math.pi*cn_sin(2*math.pi*x1),
        ]

    def hessian(self, x):
        x0, x1 = map(float, x)
        return np.diag([
            2 + 40*math.pi**2*math.cos(2*math.pi*x0),
            2 + 40*math.pi**2*math.cos(2*math.pi*x1),
        ])


class Himmelblau(Objective):
    name = "Himmelblau"
    dimension = 2
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    # Один из четырёх глобальных минимумов.
    optimum = np.array([3.0, 2.0])
    optimum_value = 0.0
    supports_derivatives = True

    def __call__(self, x):
        x0, x1 = x
        return (x0*x0 + x1 - 11)**2 + (x0 + x1*x1 - 7)**2

    def gradient(self, x):
        x0, x1 = x
        a = x0*x0 + x1 - 11
        b = x0 + x1*x1 - 7
        return [4*x0*a + 2*b, 2*a + 4*x1*b]

    def hessian(self, x):
        x0, x1 = map(float, x)
        return np.array([
            [12*x0*x0 + 4*x1 - 42, 4*x0 + 4*x1],
            [4*x0 + 4*x1, 12*x1*x1 + 4*x0 - 26],
        ], dtype=float)


class Eggholder(Objective):
    name = "Eggholder"
    dimension = 2
    bounds = [(-512.0, 512.0), (-512.0, 512.0)]
    optimum = np.array([512.0, 404.2319])
    optimum_value = -959.6407
    # Из-за abs/sqrt аналитические производные имеют особые точки; в сравнении
    # со старой лабораторной GD/Newton для этой функции не используем.
    supports_derivatives = False

    def __call__(self, x):
        x0, x1 = x
        z = x1 + 47
        term1 = -z * cn_sin(cn_sqrt(cn_abs(x0/2 + z)))
        term2 = -x0 * cn_sin(cn_sqrt(cn_abs(x0 - z)))
        return term1 + term2


class DesmosDiscontinuous(Objective):
    name = "Desmos-discontinuous"
    dimension = 2
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    # У функции несколько глобальных минимумов со значением 0.
    # Эта точка численно проверена: оба квадрата практически равны нулю.
    optimum = np.array([1.01018131, 0.81580354])
    optimum_value = 0.0
    supports_derivatives = False

    def __call__(self, x):
        x0, x1 = x
        a = cn_round(cn_sin(10*x1)) + 2
        b = cn_round(cn_sin(7*x0)) + 2
        first = (x0*a)**2 + x1 - 10
        second = x0 + (x1*b)**2 - 7
        return first**2 + second**2


OBJECTIVES = [Rastrigin2(), Himmelblau(), Eggholder(), DesmosDiscontinuous()]

STARTS = {
    "Rastrigin-2": np.array([3.7, -2.8]),
    "Himmelblau": np.array([-3.5, 3.0]),
    "Eggholder": np.array([-400.0, 200.0]),
    "Desmos-discontinuous": np.array([4.0, -3.0]),
}

SA_CONFIG = {
    "Rastrigin-2": dict(iterations=5000, temp0=10.0, cooling=0.995, sigma=0.35),
    "Himmelblau": dict(iterations=4000, temp0=10.0, cooling=0.995, sigma=0.25),
    "Eggholder": dict(iterations=5000, temp0=500.0, cooling=0.9992, sigma=20.0),
    "Desmos-discontinuous": dict(iterations=4000, temp0=20.0, cooling=0.997, sigma=0.35),
}

PSO_CONFIG = {
    "Rastrigin-2": dict(swarm=30, iterations=250, inertia=0.72, c1=1.49, c2=1.49),
    "Himmelblau": dict(swarm=30, iterations=250, inertia=0.72, c1=1.49, c2=1.49),
    "Eggholder": dict(swarm=30, iterations=250, inertia=0.72, c1=1.49, c2=1.49),
    "Desmos-discontinuous": dict(swarm=30, iterations=500, inertia=0.72, c1=1.49, c2=1.49),
}

GD_CONFIG = {
    "Rastrigin-2": dict(lr=0.01, max_iter=5000),
    "Himmelblau": dict(lr=0.01, max_iter=5000),
}
