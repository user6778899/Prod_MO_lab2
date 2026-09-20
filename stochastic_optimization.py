
from __future__ import annotations
import time
import numpy as np
from constructive_number import ConstructiveNumber
from black_boxes import Objective


def eval_cn(f: Objective, x, eps: float):
    return f.cn_value(x, eps)


def simulated_annealing(
    f: Objective, x0, bounds, iterations=5000, temp0=10.0,
    cooling=0.995, sigma=0.25, eps=1e-8, seed=67
):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    cn = eval_cn(f, x, eps)
    fx = cn.midpoint
    best_x, best_fx = x.copy(), fx

    trajectory = [x.copy()]
    f_hist = [fx]
    eps_hist = [cn.epsilon]
    accepted = 0
    calls = 1
    t0 = time.perf_counter()
    T = temp0

    for k in range(iterations):
        # Случайное возмущение.
        candidate = x + rng.normal(0.0, sigma, size=f.dimension)
        candidate = np.clip(candidate, lo, hi)

        cn_new = eval_cn(f, candidate, eps)
        f_new = cn_new.midpoint
        calls += 1

        delta = f_new - fx
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            x, fx = candidate, f_new
            accepted += 1

        if fx < best_fx:
            best_x, best_fx = x.copy(), fx

        trajectory.append(best_x.copy())
        f_hist.append(best_fx)
        eps_hist.append(eval_cn(f, best_x, eps).epsilon)
        T *= cooling

    return {
        "x": best_x, "fx": best_fx, "iterations": iterations,
        "calls": calls, "grad_calls": 0,
        "time": time.perf_counter() - t0,
        "trajectory": np.array(trajectory),
        "eps": np.array(eps_hist), "f_hist": np.array(f_hist),
        "accepted": accepted, "acceptance_rate": accepted / max(iterations, 1),
        "memory_estimate_bytes": int((x.nbytes + np.asarray(trajectory).nbytes) + 8*8),
    }


def particle_swarm(
    f: Objective, bounds, swarm=30, iterations=250,
    inertia=0.72, c1=1.49, c2=1.49, eps=1e-8, seed=67
):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    d = f.dimension

    positions = rng.uniform(lo, hi, size=(swarm, d))
    vmax = 0.2 * (hi - lo)
    velocities = rng.uniform(-vmax, vmax, size=(swarm, d))

    pbest = positions.copy()
    pbest_vals = np.empty(swarm)
    calls = 0
    t0 = time.perf_counter()

    for i in range(swarm):
        pbest_vals[i] = eval_cn(f, positions[i], eps).midpoint
        calls += 1

    gidx = int(np.argmin(pbest_vals))
    gbest = pbest[gidx].copy()
    gbest_val = float(pbest_vals[gidx])

    trajectory = [gbest.copy()]
    f_hist = [gbest_val]
    eps_hist = [eval_cn(f, gbest, eps).epsilon]

    for k in range(iterations):
        r1 = rng.random((swarm, d))
        r2 = rng.random((swarm, d))
        velocities = (
            inertia * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )
        velocities = np.clip(velocities, -vmax, vmax)
        positions = np.clip(positions + velocities, lo, hi)

        for i in range(swarm):
            val_cn = eval_cn(f, positions[i], eps)
            val = val_cn.midpoint
            calls += 1
            if val < pbest_vals[i]:
                pbest[i] = positions[i].copy()
                pbest_vals[i] = val

        gidx = int(np.argmin(pbest_vals))
        if pbest_vals[gidx] < gbest_val:
            gbest = pbest[gidx].copy()
            gbest_val = float(pbest_vals[gidx])

        trajectory.append(gbest.copy())
        f_hist.append(gbest_val)
        eps_hist.append(eval_cn(f, gbest, eps).epsilon)

    return {
        "x": gbest, "fx": gbest_val, "iterations": iterations,
        "calls": calls, "grad_calls": 0,
        "time": time.perf_counter() - t0,
        "trajectory": np.array(trajectory),
        "eps": np.array(eps_hist), "f_hist": np.array(f_hist),
        "memory_estimate_bytes": int(
            positions.nbytes + velocities.nbytes + pbest.nbytes +
            pbest_vals.nbytes + trajectory.__sizeof__() + 8*8
        ),
    }


def nelder_mead(f: Objective, x0, bounds, step=0.2, tol=1e-8, max_iter=3000, eps=1e-8):
    # Базовый метод из предыдущей лабораторной: используется для честного сравнения
    # с новым стохастическим подходом на функциях без производных.
    x0 = np.asarray(x0, dtype=float)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        y = x0.copy()
        y[i] = np.clip(y[i] + step, lo[i], hi[i])
        simplex.append(y)
    simplex = np.array(simplex)

    calls = 0
    trajectory = [x0.copy()]
    f_hist, eps_hist = [], []
    t0 = time.perf_counter()

    for it in range(max_iter):
        vals = np.array([f.value(s) for s in simplex])
        calls += n + 1
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]
        best = simplex[0]
        trajectory.append(best.copy())
        cn = eval_cn(f, best, eps)
        f_hist.append(cn.midpoint); eps_hist.append(cn.epsilon)

        if np.max(np.linalg.norm(simplex[1:] - best, axis=1)) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = np.clip(centroid + (centroid - worst), lo, hi)
        fr = f.value(xr); calls += 1

        if vals[0] <= fr < vals[-2]:
            simplex[-1] = xr
        elif fr < vals[0]:
            xe = np.clip(centroid + 2*(xr-centroid), lo, hi)
            fe = f.value(xe); calls += 1
            simplex[-1] = xe if fe < fr else xr
        else:
            if fr < vals[-1]:
                xc = np.clip(centroid + 0.5*(xr-centroid), lo, hi)
            else:
                xc = np.clip(centroid + 0.5*(worst-centroid), lo, hi)
            fc = f.value(xc); calls += 1
            if fc < vals[-1]:
                simplex[-1] = xc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5*(simplex[i]-simplex[0])

    return {
        "x": simplex[0], "fx": f.value(simplex[0]), "iterations": it+1,
        "calls": calls, "grad_calls": 0,
        "time": time.perf_counter()-t0,
        "trajectory": np.array(trajectory),
        "eps": np.array(eps_hist), "f_hist": np.array(f_hist),
        "memory_estimate_bytes": int(simplex.nbytes + np.asarray(trajectory).nbytes + 8*8),
    }
