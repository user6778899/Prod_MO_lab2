from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from black_boxes import OBJECTIVES, STARTS, SA_CONFIG, PSO_CONFIG, GD_CONFIG
from stochastic_optimization import simulated_annealing, particle_swarm, nelder_mead
from deterministic_optimization import gradient_descent, newton

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


def create_output_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _deep_size(obj, seen=None):
    """Одинаковая оценка памяти возвращаемого состояния для всех методов."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(_deep_size(v, seen) for v in obj)
    return int(size)


def run_measured(func, *args, **kwargs):
    result = func(*args, **kwargs)
    # Оценивается одним и тем же способом весь сохранённый результат метода:
    # траектория, история f, история epsilon и служебные поля.
    result["memory_bytes"] = _deep_size(result)
    return result


def run_methods(f, eps=1e-8, seed=67, fast=False):
    sa_cfg = dict(SA_CONFIG[f.name])
    pso_cfg = dict(PSO_CONFIG[f.name])
    if fast:
        sa_cfg["iterations"] = max(300, sa_cfg["iterations"] // 8)
        pso_cfg["iterations"] = max(50, pso_cfg["iterations"] // 5)

    sa = run_measured(
        simulated_annealing, f, STARTS[f.name], f.bounds,
        eps=eps, seed=seed, **sa_cfg
    )
    pso = run_measured(
        particle_swarm, f, f.bounds,
        eps=eps, seed=seed, **pso_cfg
    )
    nm = run_measured(
        nelder_mead, f, STARTS[f.name], f.bounds,
        eps=eps, max_iter=3000
    )
    return {"Simulated Annealing": sa, "Particle Swarm": pso, "Nelder-Mead": nm}


def _row(f, method, r, group, status="OK"):
    return {
        "Функция": f.name,
        "Метод": method,
        "Группа": group,
        "Статус": status,
        "Итерации": r.get("iterations", np.nan),
        "Вызовы f": r.get("calls", np.nan),
        "Вызовы ∇f": r.get("grad_calls", 0),
        "Вызовы H": r.get("hess_calls", 0),
        "Время, с": r.get("time", np.nan),
        "f(x)": r.get("fx", np.nan),
        "Ошибка f": abs(r.get("fx", np.nan) - f.optimum_value) if "fx" in r else np.nan,
        "Точка": np.array2string(r["x"], precision=6) if "x" in r else "—",
        "ε результата": float(r["eps"][-1]) if "eps" in r and len(r["eps"]) else np.nan,
        "Память сохранённого состояния, байт": r.get("memory_bytes", np.nan),
        "Acceptance rate": r.get("acceptance_rate", np.nan),
    }


def run_all(eps=1e-8):
    create_output_dirs()
    rows = []
    for f in OBJECTIVES:
        for method, r in run_methods(f, eps).items():
            group = "Лабораторная 2" if method in ("Simulated Annealing", "Particle Swarm") else "Лабораторная 1"
            rows.append(_row(f, method, r, group))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "main_results.csv", index=False, encoding="utf-8-sig")
    return df


def study_epsilon(eps_values=(1e-4, 1e-8, 1e-12)):
    rows = []
    for eps in eps_values:
        for f in OBJECTIVES:
            results = run_methods(f, eps, fast=True)
            for method in ("Simulated Annealing", "Particle Swarm"):
                r = results[method]
                rows.append({
                    "Начальная ε": eps,
                    "Функция": f.name,
                    "Метод": method,
                    "max ε": float(np.max(r["eps"])),
                    "Финальная ε": float(r["eps"][-1]),
                    "Итерации": r["iterations"],
                    "f(x)": r["fx"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "epsilon_results.csv", index=False, encoding="utf-8-sig")
    return df


def compare_with_previous(eps=1e-8):
    """Полное сравнение методов двух лабораторных.

    SA и PSO — методы второй работы. Nelder-Mead, GD и Newton — методы первой.
    GD/Newton запускаются только там, где функция имеет корректный гладкий
    градиент и Гессе. Для остальных функций записывается N/A.
    """
    create_output_dirs()

    # Если основные результаты уже посчитаны, используем их и не повторяем
    # дорогие стохастические эксперименты.
    main_path = RESULTS_DIR / "main_results.csv"
    if main_path.exists():
        base = pd.read_csv(main_path)
        rows = base.to_dict("records")
    else:
        rows = run_all(eps).to_dict("records")

    for f in OBJECTIVES:
        if f.supports_derivatives:
            gd_cfg = GD_CONFIG[f.name]
            gd_r = run_measured(
                gradient_descent, f, STARTS[f.name], f.bounds,
                eps=eps, **gd_cfg
            )
            nt_r = run_measured(
                newton, f, STARTS[f.name], f.bounds,
                eps=eps
            )
            rows.append(_row(f, "Gradient descent", gd_r, "Лабораторная 1"))
            rows.append(_row(f, "Newton", nt_r, "Лабораторная 1"))
        else:
            reason = "N/A: нет гладких производных, необходимых методу"
            for method in ("Gradient descent", "Newton"):
                rows.append({
                    "Функция": f.name,
                    "Метод": method,
                    "Группа": "Лабораторная 1",
                    "Статус": reason,
                    "Итерации": np.nan,
                    "Вызовы f": np.nan,
                    "Вызовы ∇f": np.nan,
                    "Вызовы H": np.nan,
                    "Время, с": np.nan,
                    "f(x)": np.nan,
                    "Ошибка f": np.nan,
                    "Точка": "—",
                    "ε результата": np.nan,
                    "Память сохранённого состояния, байт": np.nan,
                    "Acceptance rate": np.nan,
                })

    df = pd.DataFrame(rows)
    order = [
        "Функция", "Метод", "Группа", "Статус", "Итерации", "Вызовы f",
        "Вызовы ∇f", "Вызовы H", "Время, с", "f(x)", "Ошибка f", "Точка",
        "ε результата", "Память сохранённого состояния, байт", "Acceptance rate"
    ]
    df = df[order]
    df.to_csv(RESULTS_DIR / "methods_comparison.csv", index=False, encoding="utf-8-sig")
    return df
