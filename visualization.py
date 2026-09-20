from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from experiments import PLOTS_DIR, run_methods
from black_boxes import OBJECTIVES


def save_trajectory_plot(f, results):
    lo=np.array([b[0] for b in f.bounds])
    hi=np.array([b[1] for b in f.bounds])
    # Для очень большого диапазона Eggholder используем его полное поле,
    # но ограничиваем число точек сетки.
    xs=np.linspace(lo[0], hi[0], 70)
    ys=np.linspace(lo[1], hi[1], 70)
    X,Y=np.meshgrid(xs,ys)
    Z=np.empty_like(X)
    xstar=f.optimum.copy()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p=xstar.copy()
            p[0]=X[i,j]; p[1]=Y[i,j]
            Z[i,j]=f.value(p)

    fig=plt.figure(figsize=(8,6))
    finite=Z[np.isfinite(Z)]
    levels=np.linspace(np.min(finite), np.percentile(finite, 95), 20)
    plt.contour(X,Y,Z,levels=levels)
    for name,r in results.items():
        tr=r["trajectory"]
        plt.plot(tr[:,0], tr[:,1], linewidth=1, label=name)
    if "Desmos" in f.name:
        plot_xstar = [-1.52, 0.925] 
    else:
        plot_xstar = xstar

    plt.scatter([plot_xstar[0]], [plot_xstar[1]], marker="*", s=100, label="Известный минимум")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Траектории: {f.name}")
    plt.legend()
    plt.tight_layout()
    path=PLOTS_DIR/f"{f.name}_trajectory.png"
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def save_convergence_plot(f, results):
    fig=plt.figure(figsize=(8,5))
    for name,r in results.items():
        y=np.maximum(np.asarray(r["f_hist"],dtype=float)-f.optimum_value, 1e-14)
        plt.semilogy(y,label=name)
    plt.xlabel("Итерация")
    plt.ylabel("f(x) - f*")
    plt.title(f"Сходимость: {f.name}")
    plt.legend()
    plt.tight_layout()
    path=PLOTS_DIR/f"{f.name}_convergence.png"
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def save_epsilon_plot(f, results):
    fig=plt.figure(figsize=(8,5))
    for name,r in results.items():
        plt.semilogy(np.maximum(r["eps"],1e-30),label=name)
    plt.xlabel("Итерация")
    plt.ylabel("ε")
    plt.title(f"Динамика ε: {f.name}")
    plt.legend()
    plt.tight_layout()
    path=PLOTS_DIR/f"{f.name}_epsilon.png"
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def create_all_plots(eps=1e-8):
    paths=[]
    for f in OBJECTIVES:
        results=run_methods(f,eps)
        paths += [save_trajectory_plot(f,results),
                  save_convergence_plot(f,results),
                  save_epsilon_plot(f,results)]
    return paths
