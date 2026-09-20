from __future__ import annotations
import time
import numpy as np
from constructive_number import ConstructiveNumber


def _clip(x, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(x, lo, hi)


def _cn_point(x, eps):
    return [ConstructiveNumber.from_x_eps(float(v), eps) for v in x]


def gradient_descent(f, x0, bounds, lr=0.01, tol=1e-8, max_iter=5000, eps=1e-8):
    if not f.supports_derivatives:
        raise ValueError(f"{f.name}: derivatives are not supported")

    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    f_hist, eps_hist = [], []
    calls = 0
    grad_calls = 0
    t0 = time.perf_counter()

    for k in range(max_iter):
        cn = f.cn_value(x, eps)
        calls += 1
        f_hist.append(cn.midpoint)
        eps_hist.append(cn.epsilon)

        g_cn = f.gradient(_cn_point(x, eps))
        grad_calls += 1
        g = np.array([float(v) for v in g_cn], dtype=float)
        x_new = _clip(x - lr*g, bounds)
        trajectory.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    final = f.cn_value(x, eps)
    calls += 1
    return {
        "x": x,
        "fx": final.midpoint,
        "iterations": k + 1,
        "calls": calls,
        "grad_calls": grad_calls,
        "time": time.perf_counter() - t0,
        "trajectory": np.array(trajectory),
        "eps": np.array(eps_hist + [final.epsilon]),
        "f_hist": np.array(f_hist),
    }


def newton(f, x0, bounds, tol=1e-8, max_iter=100, eps=1e-8):
    if not f.supports_derivatives:
        raise ValueError(f"{f.name}: derivatives are not supported")

    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    f_hist, eps_hist = [], []
    calls = 0
    grad_calls = 0
    hess_calls = 0
    t0 = time.perf_counter()

    for k in range(max_iter):
        cn = f.cn_value(x, eps)
        calls += 1
        fx = cn.midpoint
        f_hist.append(fx)
        eps_hist.append(cn.epsilon)

        g_cn = f.gradient(_cn_point(x, eps))
        grad_calls += 1
        g = np.array([float(v) for v in g_cn], dtype=float)
        H = f.hessian(x)
        hess_calls += 1

        try:
            direction = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            direction = np.linalg.pinv(H) @ g

        step = 1.0
        while step > 1e-10:
            candidate = _clip(x - step*direction, bounds)
            candidate_value = f.value(candidate)
            calls += 1
            if candidate_value < fx:
                break
            step *= 0.5

        x_new = _clip(x - step*direction, bounds)
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    final = f.cn_value(x, eps)
    calls += 1
    return {
        "x": x,
        "fx": final.midpoint,
        "iterations": k + 1,
        "calls": calls,
        "grad_calls": grad_calls,
        "hess_calls": hess_calls,
        "time": time.perf_counter() - t0,
        "trajectory": np.array(trajectory),
        "eps": np.array(eps_hist + [final.epsilon]),
        "f_hist": np.array(f_hist),
    }
