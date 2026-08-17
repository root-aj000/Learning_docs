"""
generate_images.py
==================
Generates ALL matplotlib visualizations used by the four MLDL maths docs:

    calculus.md, linear-algebra.md, probability.md, statistics.md

Output directory: public/maths-images/  (served by Next.js at /maths-images/*.png)

Run:  python generate_images.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Polygon, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "public", "maths-images"))
os.makedirs(OUT, exist_ok=True)

C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GREEN = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_PURPLE = "#9467bd"
C_GRAY = "#888888"
C_LIGHT = "#c9d6e5"

DPI = 150


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def arrow(ax, start, end, color=C_BLUE, lw=2.2, label=None, style="-|>", ms=16, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=ms, color=color,
            lw=lw, linestyle=ls, zorder=5,
        )
    )
    if label:
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(mid[0] + 0.06, mid[1] + 0.06, label, fontsize=11, color=color)


# ======================================================================
# CALCULUS
# ======================================================================

def calc_slope_line():
    x = np.linspace(-1, 4, 100)
    y = 2 * x + 1
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color=C_BLUE, lw=2.5, label=r"$y = 2x + 1$")
    x1, x2 = 0.5, 3.0
    y1, y2 = 2 * x1 + 1, 2 * x2 + 1
    ax.plot([x1, x2], [y1, y1], color=C_RED, lw=2)
    ax.plot([x2, x2], [y1, y2], color=C_GREEN, lw=2)
    ax.plot([x1, x2], [y1, y2], "ko", ms=6, zorder=6)
    ax.annotate("", xy=(x2, (y1 + y2) / 2), xytext=(x1, (y1 + y2) / 2),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8))
    ax.annotate("", xy=(x2 - 0.08, y2), xytext=(x2 - 0.08, y1),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.8))
    ax.text(1.6, 3.2, "run = 2.5", color=C_RED, fontsize=11, ha="center")
    ax.text(3.15, 4.5, "rise = 5", color=C_GREEN, fontsize=11, ha="left")
    ax.text(3.3, 2.2, r"slope $= \frac{rise}{run} = \frac{5}{2.5} = 2$",
            fontsize=11, color="black", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#e0a800"))
    style_ax(ax, "x", "y", "Slope = Rise over Run (m = 2)")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.2, 4.4)
    ax.set_ylim(-0.5, 10)
    savefig(fig, "calc-slope-line.png")


def calc_secant_tangent():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.linspace(-0.2, 3.2, 300)
    f = lambda t: t ** 2
    ax.plot(x, f(x), color=C_BLUE, lw=2.5, label=r"$f(x) = x^2$")
    a = 1.0
    for h, c, lbl in [(2.0, C_GRAY, "secant h=2"), (1.0, C_ORANGE, "secant h=1"),
                      (0.5, C_PURPLE, "secant h=0.5")]:
        b = a + h
        ax.plot([a, b], [f(a), f(b)], color=c, lw=1.8, label=lbl)
    ax.plot([a], [f(a)], "o", color=C_RED, ms=8, zorder=7)
    t = np.linspace(0, 2.4, 100)
    ax.plot(t, 2 * a * t - a ** 2, color=C_RED, lw=2.6, label="tangent h→0 (slope 2a)")
    ax.annotate("f(a)", xy=(a, f(a)), xytext=(a + 0.15, f(a) + 0.9),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=11)
    style_ax(ax, "x", "y", "Secant Lines Approaching the Tangent Line")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.4, 9.5)
    savefig(fig, "calc-secant-tangent.png")


def calc_tangent():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(-0.5, 3.5, 300)
    f = lambda t: t ** 2
    a = 1.5
    ax.plot(x, f(x), color=C_BLUE, lw=2.5, label=r"$f(x) = x^2$")
    ax.plot(x, 2 * a * x - a ** 2, color=C_RED, lw=2.2, ls="--",
            label=f"tangent at x={a}: slope = 2({a}) = 3")
    ax.plot([a], [f(a)], "o", color=C_RED, ms=9, zorder=7)
    ax.annotate(f"({a}, {a**2})", xy=(a, f(a)), xytext=(a + 0.35, f(a) - 0.6),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=11)
    ax.text(2.35, 2.2, "each tiny step in x\nraises the curve ~3x", fontsize=10,
            color=C_RED, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffe4e1", ec="#d62728"))
    style_ax(ax, "x", "y", r"Tangent Line: slope = $f'(a)$")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(-0.5, 10.5)
    savefig(fig, "calc-tangent.png")


def calc_limit():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(-1.2, 3.2, 400)
    f = lambda t: (t ** 2 - 1) / (t - 1)
    y = np.where(np.abs(x - 1) < 1e-9, np.nan, f(x))
    ax.plot(x, y, color=C_BLUE, lw=2.5, label=r"$f(x) = \frac{x^2-1}{x-1} = x+1$  (x ≠ 1)")
    ax.plot([1], [2], "o", mfc="white", mec=C_RED, ms=10, zorder=7)
    ax.plot([1], [2], "o", color="white", ms=3, zorder=8)
    ax.annotate("hole at x=1\n(undefined!)", xy=(1, 2), xytext=(1.55, 2.8),
                arrowprops=dict(arrowstyle="->", color=C_RED), fontsize=10, color=C_RED)
    xs = [0.5, 0.9, 0.99, 1.5, 1.1, 1.01]
    for xi in xs:
        ax.plot([xi], [f(xi)], "o", color=C_GREEN, ms=5, zorder=6)
        ax.plot([xi, xi], [0, f(xi)], color=C_GREEN, lw=0.7, alpha=0.6)
    ax.plot([1, 1], [0, 2], color=C_GRAY, lw=1, ls=":")
    ax.text(0.62, 3.9, r"as $x \to 1^-$, $f(x) \to 2$", fontsize=10, color=C_GREEN, ha="center")
    ax.text(1.5, 3.9, r"as $x \to 1^+$, $f(x) \to 2$", fontsize=10, color=C_GREEN, ha="center")
    style_ax(ax, "x", "y", r"Limit: $\lim_{x \to 1} \frac{x^2-1}{x-1} = 2$")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.4, 3.0)
    ax.set_ylim(-0.4, 4.6)
    savefig(fig, "calc-limit.png")


def calc_partial_derivative():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    X = np.linspace(-2, 2, 60)
    Y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(X, Y)
    Z = 3 * X ** 2 + 2 * Y ** 3
    ax.plot_surface(X, Y, Z, alpha=0.85, cmap="viridis", rstride=3, cstride=3, linewidth=0)
    p = (1.0, 1.0)
    z0 = 3 * p[0] ** 2 + 2 * p[1] ** 3
    ts = np.linspace(-1.6, 1.6, 20)
    x_line = p[0] + ts
    y_line = p[1] + ts
    # tangent line in x direction (y held fixed at p[1])
    tx = p[0] + ts
    tz = z0 + 6 * p[0] * ts
    ax.plot(tx, np.full_like(tx, p[1]), tz, color=C_RED, lw=3, label=r"tangent in $x$ dir: slope $\partial f/\partial x = 6x$")
    # tangent line in y direction
    ty = p[1] + ts
    tz2 = z0 + 6 * p[1] ** 2 * ts
    ax.plot(np.full_like(ty, p[0]), ty, tz2, color=C_GREEN, lw=3, label=r"tangent in $y$ dir: slope $\partial f/\partial y = 6y^2$")
    ax.scatter([p[0]], [p[1]], [z0], color=C_BLUE, s=60, zorder=8, label="point (1,1)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(r"Partial Derivatives = Tangents Holding One Variable Fixed", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, "calc-partial-derivative.png")


def calc_gradient_contour():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    X = np.linspace(-3, 3, 200)
    Y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2
    ax = axes[0]
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    # gradient arrows at points
    for (xi, yi) in [(-2, 1), (-1, -1.5), (2, 0.5), (0.5, 2), (-0.5, 0.5), (1, -2)]:
        gx, gy = 2 * xi, 2 * yi
        n = np.sqrt(gx ** 2 + gy ** 2)
        ax.add_patch(FancyArrowPatch((xi, yi), (xi + 0.55 * gx / n, yi + 0.55 * gy / n),
                                     arrowstyle="-|>", mutation_scale=14, color=C_RED, lw=1.8))
    ax.set_title(r"$\nabla f$ points UPHILL (toward the peak)", fontsize=11)
    style_ax(ax, "x", "y", grid=False)
    ax = axes[1]
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    for (xi, yi) in [(-2, 1), (-1, -1.5), (2, 0.5), (0.5, 2), (-0.5, 0.5), (1, -2)]:
        gx, gy = 2 * xi, 2 * yi
        n = np.sqrt(gx ** 2 + gy ** 2)
        ax.add_patch(FancyArrowPatch((xi, yi), (xi - 0.55 * gx / n, yi - 0.55 * gy / n),
                                     arrowstyle="-|>", mutation_scale=14, color=C_GREEN, lw=1.8))
    ax.set_title(r"$-\nabla f$ points DOWNHILL (toward the minimum)", fontsize=11)
    style_ax(ax, "x", "y", grid=False)
    fig.suptitle("Gradient Descent uses the negative gradient as its compass", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "calc-gradient-contour.png")


def calc_concavity():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.linspace(-2.2, 2.2, 300)
    f = lambda t: t ** 2
    axes[0].plot(x, f(x), color=C_BLUE, lw=2.5)
    axes[0].set_title(r"Concave UP: $f(x)=x^2$, $f''(x)=2>0$ → MINIMUM", fontsize=10)
    style_ax(axes[0], "x", "y")
    axes[1].plot(x, -f(x), color=C_RED, lw=2.5)
    axes[1].set_title(r"Concave DOWN: $f(x)=-x^2$, $f''(x)=-2<0$ → MAXIMUM", fontsize=10)
    style_ax(axes[1], "x", "y")
    g = lambda t: t ** 3 - 3 * t
    axes[2].plot(x, g(x), color=C_PURPLE, lw=2.5, label=r"$f(x)=x^3-3x$")
    axes[2].plot(x, 3 * x ** 2 - 3, color=C_GREEN, lw=1.8, ls="--", label=r"$f'(x)=3x^2-3$")
    axes[2].plot(x, 6 * x, color=C_ORANGE, lw=1.8, ls=":", label=r"$f''(x)=6x$")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].axvline(0, color="black", lw=0.8, ls=":")
    axes[2].set_title("f, f', f'' together (inflection where f''=0)", fontsize=10)
    axes[2].legend(fontsize=8)
    style_ax(axes[2], "x", "y")
    fig.suptitle("Second Derivative = Curvature (smile bowl vs frown hill)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "calc-concavity.png")


def calc_jacobian():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # input grid
    X = np.linspace(-2, 2, 7)
    Y = np.linspace(-2, 2, 7)
    X, Y = np.meshgrid(X, Y)
    ax = axes[0]
    for (xi, yi) in zip(X.ravel(), Y.ravel()):
        ax.add_patch(FancyArrowPatch((0, 0), (xi, yi), arrowstyle="-|>",
                                     mutation_scale=10, color=C_BLUE, lw=1.3, alpha=0.85))
    ax.set_title("Input vectors x = (x₁, x₂) on a grid", fontsize=11)
    style_ax(ax, "x₁", "x₂", grid=False)
    # output: f(x,y) = (x², y²)
    fx, fy = X ** 2, Y ** 2
    ax = axes[1]
    for (xi, yi) in zip(fx.ravel(), fy.ravel()):
        ax.add_patch(FancyArrowPatch((0, 0), (xi, yi), arrowstyle="-|>",
                                     mutation_scale=10, color=C_RED, lw=1.3, alpha=0.85))
    ax.set_title(r"Output vectors $f(x,y)=(x^2, y^2)$ — the Jacobian is the local stretching", fontsize=11)
    style_ax(ax, "f₁", "f₂", grid=False)
    fig.suptitle("Jacobian: how a whole vector of outputs responds to a whole vector of inputs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "calc-jacobian.png")


def calc_hessian_surfaces():
    fig = plt.figure(figsize=(13, 4.6))
    X = np.linspace(-2, 2, 50)
    Y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(X, Y)
    cases = [
        (X ** 2 + Y ** 2, r"$z=x^2+y^2$  →  H positive definite → MIN", C_BLUE),
        (-X ** 2 - Y ** 2, r"$z=-x^2-y^2$  →  H negative definite → MAX", C_RED),
        (X ** 2 - Y ** 2, r"$z=x^2-y^2$  →  H indefinite → SADDLE", C_GREEN),
    ]
    for i, (Z, title, color) in enumerate(cases):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, alpha=0.9, color=color, rstride=2, cstride=2, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("Curvature tells us the TYPE of a critical point (∇f = 0)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "calc-hessian-surfaces.png")


def calc_convex_nonconvex():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.linspace(-2.5, 2.5, 300)
    ax = axes[0]
    f = x ** 2
    ax.plot(x, f, color=C_BLUE, lw=2.5, label=r"$f(x)=x^2$ (convex)")
    for a in [-1.5, 0.3, 1.8]:
        ax.plot(x, 2 * a * x - a ** 2, color=C_GRAY, lw=1.2, ls="--")
    ax.set_title("Convex: every tangent lies BELOW the curve\n→ 1 global minimum", fontsize=11)
    style_ax(ax, "x", "f(x)")
    ax.legend(loc="upper right")
    ax = axes[1]
    g = x ** 4 - 2 * x ** 2
    ax.plot(x, g, color=C_RED, lw=2.5, label=r"$f(x)=x^4-2x^2$ (non-convex)")
    for a in [-1.4, -0.5, 0.6, 1.6]:
        # tangent at a: g'(a)=4a^3-4a
        m = 4 * a ** 3 - 4 * a
        b = g[np.argmin(np.abs(x - a))] - m * a
        ax.plot(x, m * x + b, color=C_GRAY, lw=1.2, ls="--")
    ax.scatter([-1, 1], [-1, -1], color=C_GREEN, s=50, zorder=6, label="two local minima")
    ax.scatter([0], [0], color=C_ORANGE, s=50, zorder=6, label="local max (middle bump)")
    ax.set_title("Non-convex: multiple valleys and hills\n→ many local minima, no guarantee", fontsize=11)
    style_ax(ax, "x", "f(x)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    savefig(fig, "calc-convex-nonconvex.png")


def calc_gd_1d():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.linspace(-4, 4, 300)
    f = (x - 1) ** 2 + 2
    ax.plot(x, f, color=C_BLUE, lw=2.5, label=r"$L(w)=(w-1)^2+2$")
    w = 3.0
    alpha = 0.25
    pts = []
    for _ in range(8):
        pts.append((w, (w - 1) ** 2 + 2))
        grad = 2 * (w - 1)
        w = w - alpha * grad
    pts.append((w, (w - 1) ** 2 + 2))
    for i, (wi, fi) in enumerate(pts):
        ax.plot([wi], [fi], "o", color=C_RED, ms=7, zorder=6)
        ax.annotate(str(i), (wi, fi), textcoords="offset points", xytext=(6, 6), fontsize=9, color=C_RED)
    ax.axhline(2, color=C_GREEN, lw=1, ls=":", label="global minimum (w=1)")
    ax.set_title("Gradient Descent in 1D: stepping downhill toward w=1", fontsize=12, fontweight="bold")
    style_ax(ax, "w", "Loss L(w)")
    ax.legend(loc="upper right")
    savefig(fig, "calc-gd-1d.png")


def calc_gd_2d():
    fig, ax = plt.subplots(figsize=(7, 6))
    X = np.linspace(-4, 4, 200)
    Y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(X, Y)
    Z = (X - 0.5) ** 2 + 1.2 * (Y + 0.5) ** 2
    ax.contour(X, Y, Z, levels=25, cmap="viridis")
    w = np.array([-3.5, 2.8])
    alpha = 0.18
    path = [w.copy()]
    for _ in range(25):
        grad = np.array([2 * (w[0] - 0.5), 2.4 * (w[1] + 0.5)])
        w = w - alpha * grad
        path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=C_RED, lw=1.8, zorder=5)
    ax.plot(path[:, 0], path[:, 1], "o", color=C_RED, ms=3.5, zorder=6)
    ax.scatter([0.5], [-0.5], color=C_GREEN, s=90, marker="*", zorder=7, label="minimum")
    ax.set_title("Gradient Descent in 2D: following −∇L on a contour map", fontsize=12, fontweight="bold")
    style_ax(ax, "w₁", "w₂", grid=False)
    ax.legend()
    savefig(fig, "calc-gd-2d.png")


def calc_learning_rate():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.linspace(-3, 3, 300)
    f = x ** 2
    def run(alpha, n=30, w0=2.2):
        w = w0
        pts = []
        for _ in range(n):
            pts.append((w, w ** 2))
            w = w - alpha * 2 * w
        return pts
    ax = axes[0]
    pts = run(0.05, 60)
    ax.plot(x, f, color=C_BLUE, lw=2)
    ax.plot(*zip(*pts), color=C_RED, lw=1.5)
    ax.set_title("α = 0.05 (too SMALL): creeps slowly, many steps", fontsize=10)
    style_ax(ax, "w", "loss")
    ax = axes[1]
    pts = run(0.3)
    ax.plot(x, f, color=C_BLUE, lw=2)
    ax.plot(*zip(*pts), color=C_GREEN, lw=1.5)
    ax.set_title("α = 0.3 (just RIGHT): converges quickly", fontsize=10)
    style_ax(ax, "w", "loss")
    ax = axes[2]
    pts = run(1.1, 12)
    ax.plot(x, f, color=C_BLUE, lw=2)
    ax.plot(*zip(*pts), color=C_ORANGE, lw=1.5)
    ax.set_title("α = 1.1 (too LARGE): overshoots, diverges!", fontsize=10)
    style_ax(ax, "w", "loss")
    fig.suptitle("The Learning Rate α controls the size of each step", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "calc-learning-rate.png")


def calc_momentum():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    X = np.linspace(-4, 4, 200)
    Y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(X, Y)
    Z = 0.5 * X ** 2 + 8 * Y ** 2
    ax.contour(X, Y, Z, levels=20, cmap="viridis")
    # plain GD: zigzag
    w = np.array([-3.5, 2.2])
    alpha = 0.05
    path = [w.copy()]
    for _ in range(60):
        grad = np.array([w[0], 16 * w[1]])
        w = w - alpha * grad
        path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=C_ORANGE, lw=1.8, label="plain GD (zigzag)")
    # momentum GD
    w = np.array([-3.5, 2.2])
    v = np.zeros(2)
    beta, alpha = 0.9, 0.05
    path = [w.copy()]
    for _ in range(60):
        grad = np.array([w[0], 16 * w[1]])
        v = beta * v + (1 - beta) * grad
        w = w - alpha * v
        path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=C_GREEN, lw=2.2, label="momentum GD (smooth)")
    ax.scatter([0], [0], color=C_BLUE, s=90, marker="*", zorder=7, label="minimum")
    ax.set_title("Momentum: average of past gradients smooths the zigzag", fontsize=12, fontweight="bold")
    style_ax(ax, "w₁", "w₂", grid=False)
    ax.legend(loc="upper right")
    savefig(fig, "calc-momentum.png")


def calc_riemann():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.linspace(-0.2, 2.2, 300)
    f = lambda t: t ** 2
    ax.plot(x, f(x), color=C_BLUE, lw=2.5, label=r"$f(x)=x^2$")
    n = 8
    a, b = 0, 2
    xs = np.linspace(a, b, n + 1)
    for i in range(n):
        ax.add_patch(Rectangle((xs[i], 0), xs[i + 1] - xs[i], f(xs[i + 1]),
                               facecolor=C_RED, alpha=0.35, edgecolor=C_RED, lw=1))
    ax.set_title("Riemann sum: approximating the area with rectangles", fontsize=12, fontweight="bold")
    style_ax(ax, "x", "y")
    ax.legend(loc="upper left")
    savefig(fig, "calc-riemann.png")


def calc_definite_integral():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.linspace(-0.5, 3, 400)
    f = lambda t: -t ** 2 + 3 * t + 1
    ax.plot(x, f(x), color=C_BLUE, lw=2.5, label=r"$f(x)=-x^2+3x+1$")
    a, b = 0.5, 2.5
    xs = np.linspace(a, b, 100)
    ax.fill_between(xs, f(xs), color=C_GREEN, alpha=0.4)
    ax.axvline(a, color=C_RED, lw=1.5, ls="--")
    ax.axvline(b, color=C_RED, lw=1.5, ls="--")
    ax.text((a + b) / 2, 1.0, r"Area = $\int_a^b f(x)\,dx$", fontsize=13, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#e0a800"))
    ax.annotate("a", xy=(a, 0), xytext=(a, -0.7), fontsize=12, ha="center")
    ax.annotate("b", xy=(b, 0), xytext=(b, -0.7), fontsize=12, ha="center")
    ax.set_title("Definite integral = total area under the curve between a and b",
                 fontsize=12, fontweight="bold")
    style_ax(ax, "x", "f(x)")
    ax.legend(loc="upper right")
    ax.set_ylim(-1, 4)
    savefig(fig, "calc-definite-integral.png")


def calc_gaussian_pdf():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mu, sigma = 2.0, 0.8
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, color=C_BLUE, lw=2.5, label=r"PDF $f(x)$")
    a, b = 1.0, 2.6
    xs = np.linspace(a, b, 100)
    ax.fill_between(xs, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
                    color=C_GREEN, alpha=0.5)
    ax.axvline(a, color=C_RED, lw=1.2, ls="--")
    ax.axvline(b, color=C_RED, lw=1.2, ls="--")
    ax.annotate("", xy=(a, 0.30), xytext=(b, 0.30),
                arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.8))
    ax.text((a + b) / 2, 0.33, r"$P(a \leq X \leq b) = \int_a^b f(x)\,dx$",
            fontsize=12, ha="center", color=C_GREEN)
    ax.axvline(mu, color=C_ORANGE, lw=1.5, ls=":", label=r"mean $\mu$")
    ax.set_title("Probability of an interval = area under the PDF", fontsize=12, fontweight="bold")
    style_ax(ax, "x", "density f(x)")
    ax.legend(loc="upper right")
    savefig(fig, "calc-gaussian-pdf.png")


def calc_diffusion_noise():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    t = np.linspace(0, 4, 300)
    clean = np.sin(2 * np.pi * t)
    rng = np.random.default_rng(42)
    for i, (sigma, title) in enumerate([(0.05, "t=0  clean signal"), (0.6, "t=5  a little noise"),
                                        (1.8, "t=10  heavily noised")]):
        axes[i].plot(t, clean + rng.normal(0, sigma, t.size), color=C_BLUE, lw=1.2)
        axes[i].plot(t, clean, color=C_GRAY, lw=0.8, ls=":")
        axes[i].set_title(title, fontsize=10)
        style_ax(axes[i], "time", "value", grid=False)
    fig.suptitle("Diffusion: adding increasing noise step-by-step (forward process)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "calc-diffusion-noise.png")


# ======================================================================
# LINEAR ALGEBRA
# ======================================================================

def linalg_vector_basics():
    fig, ax = plt.subplots(figsize=(7, 6))
    v = (3, 2)
    arrow(ax, (0, 0), v, color=C_BLUE, label=r"$\mathbf{x}=(3,2)$")
    ax.plot([0, 3], [0, 0], color=C_RED, lw=2)
    ax.plot([3, 3], [0, 2], color=C_GREEN, lw=2)
    ax.text(1.2, -0.35, "x₁ = 3", color=C_RED, fontsize=11)
    ax.text(3.1, 1.0, "x₂ = 2", color=C_GREEN, fontsize=11)
    ax.scatter([3], [2], color=C_BLUE, s=40, zorder=7)
    ax.set_xlim(-0.8, 4.5); ax.set_ylim(-1, 3.6)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x₁", "x₂", "A vector = an arrow with magnitude (length) and direction", grid=False)
    savefig(fig, "linalg-vector-basics.png")


def linalg_vector_add():
    fig, ax = plt.subplots(figsize=(7, 6))
    a = (2, 1)
    b = (1, 2)
    arrow(ax, (0, 0), a, color=C_BLUE, label=r"$\mathbf{a}$")
    arrow(ax, (0, 0), b, color=C_GREEN, label=r"$\mathbf{b}$")
    arrow(ax, a, (a[0] + b[0], a[1] + b[1]), color=C_GREEN, lw=1.6, ls="--")
    arrow(ax, b, (a[0] + b[0], a[1] + b[1]), color=C_BLUE, lw=1.6, ls="--")
    arrow(ax, (0, 0), (a[0] + b[0], a[1] + b[1]), color=C_RED, lw=2.6, label=r"$\mathbf{a}+\mathbf{b}=(3,3)$")
    ax.add_patch(Polygon([(0, 0), a, (3, 3), b], closed=True, fill=True,
                         facecolor=C_LIGHT, edgecolor=C_GRAY, lw=1))
    ax.set_xlim(-0.8, 4); ax.set_ylim(-0.8, 4)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y", "Vector addition = parallelogram rule (tip-to-tail)", grid=False)
    ax.legend(loc="upper left", fontsize=10)
    savefig(fig, "linalg-vector-add.png")


def linalg_vector_scale():
    fig, ax = plt.subplots(figsize=(7, 6))
    v = (1.5, 1.0)
    arrow(ax, (0, 0), v, color=C_BLUE, label=r"$\mathbf{v}$")
    arrow(ax, (0, 0), (3.0, 2.0), color=C_RED, label=r"$2\mathbf{v}$")
    arrow(ax, (0, 0), (-0.75, -0.5), color=C_GREEN, label=r"$-0.5\mathbf{v}$")
    ax.set_xlim(-1.6, 4.2); ax.set_ylim(-1.6, 3.2)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y",
             "Scalar multiplication: same direction, new length (negative flips direction)", grid=False)
    ax.legend(loc="upper left", fontsize=10)
    savefig(fig, "linalg-vector-scale.png")


def linalg_dot_product():
    fig, ax = plt.subplots(figsize=(7, 6))
    a = (3.2, 1.6)
    b = (3.8, 0)
    arrow(ax, (0, 0), a, color=C_BLUE, label=r"$\mathbf{a}$")
    arrow(ax, (0, 0), b, color=C_RED, label=r"$\mathbf{b}$")
    theta = np.arctan2(a[1], a[0])
    ax.add_patch(Wedge((0, 0), 1.0, 0, np.degrees(theta), width=0.18, facecolor=C_ORANGE, edgecolor="none"))
    ax.text(0.62, 0.22, r"$\theta$", fontsize=13, color=C_ORANGE)
    # projection of a onto b
    proj_len = (a[0] * b[0]) / np.linalg.norm(b)
    proj = (proj_len * b[0] / np.linalg.norm(b), 0)
    ax.plot([a[0], proj[0]], [a[1], proj[1]], color=C_GREEN, lw=1.6, ls="--")
    ax.plot([proj[0], proj[0]], [0, a[1]], color=C_GREEN, lw=1.2, ls=":")
    ax.text(1.05, 0.95, r"$\|\mathbf{a}\|\cos\theta$", fontsize=12, color=C_GREEN)
    ax.set_xlim(-0.8, 4.6); ax.set_ylim(-0.8, 2.8)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y",
             r"$\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ = how much they align",
             grid=False)
    ax.legend(loc="upper right", fontsize=10)
    savefig(fig, "linalg-dot-product.png")


def linalg_norms():
    fig, ax = plt.subplots(figsize=(7, 6))
    v = (-3, 4)
    arrow(ax, (0, 0), v, color=C_BLUE, label=r"$\mathbf{x}=(-3,4)$")
    # L1 path
    ax.plot([0, -3], [0, 0], color=C_RED, lw=2, label="L1 path (|x₁| + |x₂| = 7)")
    ax.plot([-3, -3], [0, 4], color=C_RED, lw=2)
    # L2 straight line
    ax.plot([0, -3], [0, 4], color=C_GREEN, lw=2.2, ls="--", label=r"L2 straight line ($\sqrt{3^2+4^2}=5$)")
    ax.text(-1.6, -0.45, "4", color=C_RED, fontsize=11, ha="center")
    ax.text(-3.2, 2.2, "3", color=C_RED, fontsize=11)
    ax.text(-1.55, 2.3, "5", color=C_GREEN, fontsize=11)
    ax.set_xlim(-4.4, 1); ax.set_ylim(-1, 5)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y", "L1 vs L2 norm: city-block route vs straight-line route", grid=False)
    ax.legend(loc="upper right", fontsize=10)
    savefig(fig, "linalg-norms.png")


def linalg_unit_circles():
    fig, ax = plt.subplots(figsize=(7, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=C_GREEN, lw=2.5, label="L2 unit circle")
    diamond = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)]
    dx, dy = zip(*diamond)
    ax.plot(dx, dy, color=C_RED, lw=2.5, label="L1 unit diamond")
    square = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
    sx, sy = zip(*square)
    ax.plot(sx, sy, color=C_PURPLE, lw=2.5, label="L∞ unit square")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    style_ax(ax, "x₁", "x₂", "Unit circles of different norms (all points with ‖x‖ = 1)", grid=False)
    ax.legend(loc="upper right", fontsize=10)
    savefig(fig, "linalg-unit-circles.png")


def linalg_cosine_similarity():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    cases = [
        ((3, 1.6), (3, 1.2), r"$\theta \approx 0$: cos ≈ 1 (same direction)", C_GREEN),
        ((3, 0), (0, 2.4), r"$\theta = 90°$: cos = 0 (orthogonal)", C_ORANGE),
        ((3, 1.4), (-3, -1.4), r"$\theta = 180°$: cos = −1 (opposite)", C_RED),
    ]
    for i, (a, b, title, c) in enumerate(cases):
        ax = axes[i]
        arrow(ax, (0, 0), a, color=C_BLUE, label=r"$\mathbf{a}$")
        arrow(ax, (0, 0), b, color=C_RED, label=r"$\mathbf{b}$")
        ax.set_title(title, fontsize=10)
        lim = 4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.8, lim * 0.8)
        ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
        style_ax(ax, "", "", grid=False)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Cosine similarity measures the ANGLE, ignoring vector length", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-cosine-similarity.png")


def linalg_cross_product():
    fig, ax = plt.subplots(figsize=(7, 6))
    a = (3, 1)
    b = (1, 2.5)
    arrow(ax, (0, 0), a, color=C_BLUE, label=r"$\mathbf{a}$")
    arrow(ax, (0, 0), b, color=C_GREEN, label=r"$\mathbf{b}$")
    ax.add_patch(Polygon([(0, 0), a, (a[0] + b[0], a[1] + b[1]), b], closed=True,
                         facecolor=C_LIGHT, edgecolor=C_RED, lw=2))
    det = a[0] * b[1] - a[1] * b[0]
    ax.text(1.4, 1.6, f"area = |det| = {det:.1f}", fontsize=12, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#e0a800"))
    ax.set_xlim(-0.5, 4.6); ax.set_ylim(-0.5, 4.2)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y",
             r"In 2D, the cross-product magnitude = parallelogram area $a_1b_2 - a_2b_1$", grid=False)
    ax.legend(loc="upper left", fontsize=10)
    savefig(fig, "linalg-cross-product.png")


def linalg_matrix_vector():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cols = [(2, 0.5), (1, 2)]
    arrow(ax, (0, 0), cols[0], color=C_RED, label=r"$c_1$ = column 1")
    arrow(ax, (0, 0), cols[1], color=C_GREEN, label=r"$c_2$ = column 2")
    x = (2, 1.5)
    res = x[0] * np.array(cols[0]) + x[1] * np.array(cols[1])
    arrow(ax, (0, 0), x[0] * np.array(cols[0]), color=C_RED, lw=1.4, ls="--")
    arrow(ax, x[0] * np.array(cols[0]), res, color=C_GREEN, lw=1.4, ls="--")
    arrow(ax, (0, 0), res, color=C_BLUE, lw=2.6,
          label=r"$A\mathbf{x} = 2c_1 + 1.5c_2$")
    ax.text(2.0, 1.9, r"$\mathbf{x}=(2,1.5)$", fontsize=11)
    ax.set_xlim(-0.5, 6.2); ax.set_ylim(-0.5, 5.2)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "", "", "Matrix × vector = weighted combination of the COLUMNS", grid=False)
    ax.legend(loc="upper left", fontsize=10)
    savefig(fig, "linalg-matrix-vector.png")


def linalg_matmul():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    C = A @ B
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis("off")
    # draw A (2x3)
    ax.text(0.5, 4.7, "A (2×3)", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(3):
            ax.add_patch(Rectangle((0.3 + j, 4 - i), 0.8, 0.8, facecolor="white",
                                   edgecolor="black", lw=1.2))
            ax.text(0.7 + j, 4.4 - i, str(A[i, j]), ha="center", va="center", fontsize=11)
    # draw B (3x2)
    ax.text(3.3, 4.7, "B (3×2)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((3.1 + j, 4 - i), 0.8, 0.8, facecolor="white",
                                   edgecolor="black", lw=1.2))
            ax.text(3.5 + j, 4.4 - i, str(B[i, j]), ha="center", va="center", fontsize=11)
    # draw C (2x2)
    ax.text(7.0, 4.7, "C = AB (2×2)", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((6.8 + j, 4 - i), 1.1, 0.8, facecolor="#e8f5e9",
                                   edgecolor=C_GREEN, lw=1.6))
            ax.text(7.35 + j, 4.4 - i, str(C[i, j]), ha="center", va="center", fontsize=11)
    # arrows
    ax.annotate("", xy=(3.4, 3.4), xytext=(2.0, 3.4),
                arrowprops=dict(arrowstyle="->", color=C_GRAY))
    ax.annotate("", xy=(6.6, 3.4), xytext=(5.4, 3.4),
                arrowprops=dict(arrowstyle="->", color=C_GRAY))
    ax.text(5.1, 4.1, "×", fontsize=16, color=C_GRAY)
    ax.text(0.4, 0.9,
            r"$C_{11}$ = row₁·col₁ = 1·7+2·9+3·11 = 58" + "\n"
            r"$C_{12}$ = row₁·col₂ = 1·8+2·10+3·12 = 64" + "\n"
            r"$C_{21}$ = row₂·col₁ = 4·7+5·9+6·11 = 139" + "\n"
            r"$C_{22}$ = row₂·col₂ = 4·8+5·10+6·12 = 154",
            fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#e0a800"))
    ax.set_title("Matrix multiplication: dot product of each ROW of A with each COLUMN of B",
                 fontsize=12, fontweight="bold")
    savefig(fig, "linalg-matmul.png")


def linalg_transpose():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, M, title in [(axes[0], A, r"$A$ (2×3)"), (axes[1], A.T, r"$A^T$ (3×2)")]:
        ax.axis("off")
        n, m = M.shape
        for i in range(n):
            for j in range(m):
                ax.add_patch(Rectangle((j, n - 1 - i), 0.9, 0.9, facecolor="white",
                                       edgecolor="black", lw=1.2))
                ax.text(j + 0.45, n - 0.55 - i, str(M[i, j]), ha="center", va="center", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(-0.5, m + 0.3); ax.set_ylim(-0.5, n + 0.3)
        ax.text(m / 2 - 0.3, -0.35, f"{n} rows × {m} cols", fontsize=11, ha="center")
    fig.suptitle("Transpose: swap rows ↔ columns (mirror across the diagonal)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-transpose.png")


def _square(ax, pts, color, alpha=0.8, label=None, edge="black", lw=1.5):
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=alpha,
                         edgecolor=edge, lw=lw, label=label))


def linalg_det_area():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    unit = [(0, 0), (1, 0), (1, 1), (0, 1)]
    M = np.array([[2, 0], [0, 3]])
    ax = axes[0]
    _square(ax, unit, C_LIGHT, label="unit square (area 1)")
    tpts = [(M @ np.array(p)).tolist() for p in unit]
    _square(ax, tpts, C_GREEN, alpha=0.7, label="after scale by (2, 3)")
    ax.set_title(r"det = area scaling factor: det(diag(2, 3)) = 2×3 = 6", fontsize=11)
    style_ax(ax, "x", "y", grid=False)
    ax.set_xlim(-0.4, 2.8); ax.set_ylim(-0.4, 3.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    M = np.array([[1, 2], [2, 4]])
    ax = axes[1]
    _square(ax, unit, C_LIGHT)
    tpts = [(M @ np.array(p)).tolist() for p in unit]
    _square(ax, tpts, C_RED, alpha=0.7)
    ax.set_title(r"det = 0: matrix squashes 2D into a LINE (rank 1)", fontsize=11)
    style_ax(ax, "x", "y", grid=False)
    ax.set_xlim(-0.4, 5.8); ax.set_ylim(-0.4, 6.8)
    ax.set_aspect("equal")
    fig.suptitle("Determinant = how much the transformation stretches (or squashes) area",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-det-area.png")


def linalg_transformations():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    unit = [(0, 0), (1, 0), (1, 1), (0, 1)]
    mats = [
        (np.array([[1, 0], [0, 1]]), "identity", C_GRAY),
        (np.array([[2, 0], [0, 0.5]]), "scale x2, squash y÷2", C_BLUE),
        (np.array([[0, -1], [1, 0]]), "rotate 90°", C_GREEN),
        (np.array([[1, 0.7], [0, 1]]), "shear", C_ORANGE),
    ]
    for ax, (M, title, c) in zip(axes.ravel(), mats):
        _square(ax, unit, C_LIGHT, alpha=0.6)
        tpts = [(M @ np.array(p)).tolist() for p in unit]
        _square(ax, tpts, c, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.6, 3.2); ax.set_ylim(-1.6, 3.2)
        ax.set_aspect("equal")
        style_ax(ax, "x", "y", grid=False)
    fig.suptitle("Linear transformations of the unit square (origin stays fixed, lines stay straight)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "linalg-transformations.png")


def linalg_rank():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    arrow(ax, (0, 0), (2, 1), color=C_BLUE, label=r"$\mathbf{v}_1=(2,1)$")
    arrow(ax, (0, 0), (4, 2), color=C_RED, label=r"$\mathbf{v}_2=(4,2)=2\mathbf{v}_1$")
    ax.set_title("Dependent: v₂ = 2·v₁ (both on ONE line) → rank 1", fontsize=10.5)
    ax.set_xlim(-0.6, 5); ax.set_ylim(-0.8, 3)
    style_ax(ax, "", "", grid=False)
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    arrow(ax, (0, 0), (3, 1), color=C_BLUE, label=r"$\mathbf{v}_1=(3,1)$")
    arrow(ax, (0, 0), (1, 3), color=C_GREEN, label=r"$\mathbf{v}_2=(1,3)$")
    ax.set_title("Independent: not on the same line → rank 2", fontsize=10.5)
    ax.set_xlim(-0.6, 4.2); ax.set_ylim(-0.8, 3.8)
    style_ax(ax, "", "", grid=False)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Rank = number of independent directions the vectors cover", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-rank.png")


def linalg_span():
    fig, ax = plt.subplots(figsize=(7, 6))
    v1 = (3, 1)
    v2 = (1, 2)
    corners = [np.array(v1) + np.array(v2), -np.array(v1) + np.array(v2),
               -np.array(v1) - np.array(v2), np.array(v1) - np.array(v2)]
    ax.add_patch(Polygon(corners, closed=True, facecolor=C_LIGHT, alpha=0.7,
                         edgecolor=C_PURPLE, lw=1.5, label="span{v₁, v₂} = whole plane"))
    arrow(ax, (0, 0), v1, color=C_BLUE, label=r"$\mathbf{v}_1$")
    arrow(ax, (0, 0), v2, color=C_GREEN, label=r"$\mathbf{v}_2$")
    ax.set_title("Two independent vectors SPAN the entire 2D plane", fontsize=11)
    ax.set_xlim(-5, 5); ax.set_ylim(-4.5, 4.5)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y", grid=False)
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, "linalg-span.png")


def linalg_linear_system():
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.linspace(-1, 4.5, 100)
    ax.plot(x, 3 - x, color=C_BLUE, lw=2.5, label=r"$x + y = 3$")
    ax.plot(x, x - 1, color=C_GREEN, lw=2.5, label=r"$x - y = 1$")
    ax.scatter([2], [1], color=C_RED, s=90, zorder=8, label="solution (2, 1)")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_title("A 2×2 linear system = two lines; the solution is where they cross", fontsize=11)
    style_ax(ax, "x", "y", grid=False)
    ax.legend(loc="upper right")
    ax.set_xlim(-1, 4.5); ax.set_ylim(-1, 4.5)
    savefig(fig, "linalg-linear-system.png")


def linalg_eigenvectors():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    # eigenvalues 3 and 1, eigenvectors (1,1) and (1,-1)
    eigs, vecs = np.linalg.eigh(A)
    v1 = vecs[:, 0]  # (1,-1)/sqrt2
    v2 = vecs[:, 1]  # (1,1)/sqrt2
    scale = 2.2
    arrow(ax, (0, 0), scale * v2, color=C_GREEN, label=r"$\mathbf{v}=(1,1)$, eigenvalue λ=3")
    arrow(ax, (0, 0), scale * v1, color=C_ORANGE, label=r"$\mathbf{v}=(1,-1)$, eigenvalue λ=1")
    arrow(ax, (0, 0), A @ (scale * v2), color=C_GREEN, lw=2.2, ls="--",
          label=r"$A\mathbf{v}$ (same direction, 3× longer)")
    arrow(ax, (0, 0), A @ (scale * v1), color=C_ORANGE, lw=2.2, ls="--",
          label=r"$A\mathbf{v}$ (same direction, same length)")
    ax.set_title(r"$A\mathbf{v} = \lambda\mathbf{v}$: eigenvector direction unchanged, scaled by λ",
                 fontsize=11)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    style_ax(ax, "x", "y", grid=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    savefig(fig, "linalg-eigenvectors.png")


def linalg_pca():
    rng = np.random.default_rng(7)
    cov = np.array([[3.2, 2.2], [2.2, 2.0]])
    data = rng.multivariate_normal([0, 0], cov, 300)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pc1, pc2 = eigvecs[:, order[0]], eigvecs[:, order[1]]
    l1, l2 = eigvals[order[0]], eigvals[order[1]]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(data[:, 0], data[:, 1], s=18, alpha=0.6, color=C_BLUE, label="data points")
    for ev, lam, c, lbl in [(pc1, l1, C_RED, f"PC1 (λ={l1:.2f})"), (pc2, l2, C_GREEN, f"PC2 (λ={l2:.2f})")]:
        arr = 3 * np.sqrt(lam) * ev
        arrow(ax, (0, 0), arr, color=c, lw=2.6, label=lbl)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_title("PCA: eigenvectors of the covariance matrix point along max-variance directions",
                 fontsize=11)
    style_ax(ax, "feature 1", "feature 2", grid=False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    savefig(fig, "linalg-pca.png")


def linalg_svd():
    A = np.array([[2.0, 1.0], [0.5, 1.5]])
    U, s, Vt = np.linalg.svd(A)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    unit = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # Panel 1: input unit circle
    ax = axes[0]
    ax.plot(circle[0], circle[1], color=C_BLUE, lw=2, label="unit circle")
    _square(ax, unit, C_LIGHT, alpha=0.5)
    ax.set_title("1) input: unit circle", fontsize=11)
    style_ax(ax, "", "", grid=False)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.legend(fontsize=9)
    # Panel 2: V^T rotation
    ax = axes[1]
    rot = Vt @ circle
    pts = [(Vt @ np.array(p)).tolist() for p in unit]
    _square(ax, pts, C_GREEN, alpha=0.5)
    ax.plot(rot[0], rot[1], color=C_GREEN, lw=2, label="after Vᵀ (rotation)")
    ax.set_title("2) rotate by Vᵀ", fontsize=11)
    style_ax(ax, "", "", grid=False)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.legend(fontsize=9)
    # Panel 3: scale then rotate by U
    ax = axes[2]
    final = U @ np.diag(s) @ Vt @ circle
    pts = [(A @ np.array(p)).tolist() for p in unit]
    _square(ax, pts, C_RED, alpha=0.5)
    ax.plot(final[0], final[1], color=C_RED, lw=2, label="after Σ (scale) then U (rotation)")
    ax.set_title("3) result: ellipse (SVD = rotate → scale → rotate)", fontsize=11)
    style_ax(ax, "", "", grid=False)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.suptitle("SVD: A = U Σ Vᵀ turns a circle into an ellipse", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-svd.png")


def linalg_tensors():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    ax = axes[0]
    ax.scatter([0], [0], color=C_BLUE, s=150, zorder=6)
    ax.set_title("0D scalar: 5", fontsize=11)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    style_ax(ax, "", "", grid=False)
    ax = axes[1]
    arrow(ax, (0, 0), (1, 0.7), color=C_BLUE, lw=2.5, label="[2, 3]")
    ax.set_title("1D vector: [2, 3]", fontsize=11)
    ax.set_xlim(-1.6, 1.8); ax.set_ylim(-1.2, 1.6)
    style_ax(ax, "", "", grid=False)
    ax.legend(fontsize=9, loc="upper right")
    ax = axes[2]
    M = np.array([[1, 2], [3, 4]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor="white", edgecolor=C_BLUE, lw=1.4))
            ax.text(j + 0.5, 0.5 - i, str(M[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("2D matrix: 2×2 grid", fontsize=11)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.6, 2.4)
    style_ax(ax, "", "", grid=False)
    ax = axes[3]
    r = 1.0
    for z in [0, 0.45, 0.9]:
        ax.add_patch(Polygon([(0.5 + z, 0.2 + z), (1.5 + z, 0.2 + z), (1.5 + z, 1.2 + z),
                              (0.5 + z, 1.2 + z)], closed=True, fill=True,
                             facecolor=C_LIGHT, alpha=0.8, edgecolor=C_BLUE, lw=1.3))
        ax.add_patch(Polygon([(0.5 + z, 0.2 + z), (0.5 + z + 0.35, 0.55 + z), (0.5 + z, 1.2 + z)],
                             closed=True, fill=True, facecolor=C_LIGHT, alpha=0.6,
                             edgecolor=C_BLUE, lw=1.0))
        ax.add_patch(Polygon([(1.5 + z, 0.2 + z), (1.5 + z + 0.35, 0.55 + z), (1.5 + z, 1.2 + z)],
                             closed=True, fill=True, facecolor=C_LIGHT, alpha=0.6,
                             edgecolor=C_BLUE, lw=1.0))
    ax.set_title("3D tensor: cube stack (e.g. RGB image W×H×3)", fontsize=11)
    ax.set_xlim(-0.4, 3.2); ax.set_ylim(-0.4, 2.6)
    style_ax(ax, "", "", grid=False)
    fig.suptitle("Tensors: generalization of scalar → vector → matrix → higher dimensions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "linalg-tensors.png")


# ======================================================================
# PROBABILITY
# ======================================================================

def prob_venn():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    def venn(ax, shade=None, title=""):
        ax.add_patch(Circle((0.5, 0.5), 0.3, fill=False, edgecolor=C_BLUE, lw=2))
        ax.add_patch(Circle((0.78, 0.5), 0.3, fill=False, edgecolor=C_GREEN, lw=2))
        ax.text(0.28, 0.5, "A", fontsize=13, fontweight="bold", color=C_BLUE)
        ax.text(0.99, 0.5, "B", fontsize=13, fontweight="bold", color=C_GREEN)
        ax.set_xlim(0, 1.3); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10.5)
    ax = axes[0, 0]
    venn(ax, title=r"$A \cap B$: both A AND B")
    ax.add_patch(Circle((0.5, 0.5), 0.3, fill=False, edgecolor=C_BLUE, lw=2))
    ax.add_patch(Circle((0.78, 0.5), 0.3, fill=False, edgecolor=C_GREEN, lw=2))
    ax.add_patch(Circle((0.64, 0.5), 0.2, facecolor=C_RED, alpha=0.55, edgecolor="none"))
    ax.text(0.64, 0.5, "A∩B", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.text(0.28, 0.5, "A", fontsize=13, fontweight="bold", color=C_BLUE)
    ax.text(0.99, 0.5, "B", fontsize=13, fontweight="bold", color=C_GREEN)
    ax = axes[0, 1]
    venn(ax, title=r"$A \cup B$: A OR B (or both)")
    ax.add_patch(Circle((0.5, 0.5), 0.3, facecolor=C_RED, alpha=0.5, edgecolor=C_BLUE, lw=2))
    ax.add_patch(Circle((0.78, 0.5), 0.3, facecolor=C_RED, alpha=0.5, edgecolor=C_GREEN, lw=2))
    ax.text(0.28, 0.5, "A", fontsize=13, fontweight="bold", color=C_BLUE)
    ax.text(0.99, 0.5, "B", fontsize=13, fontweight="bold", color=C_GREEN)
    ax = axes[1, 0]
    venn(ax, title=r"$A^c$: complement (NOT A)")
    ax.add_patch(Rectangle((0, 0), 1.3, 1, facecolor=C_RED, alpha=0.4))
    ax.add_patch(Circle((0.5, 0.5), 0.3, facecolor="white", edgecolor=C_BLUE, lw=2))
    ax.text(0.28, 0.5, "A", fontsize=13, fontweight="bold", color=C_BLUE)
    ax.text(0.78, 0.9, "everything outside A", fontsize=9, color=C_RED, ha="center")
    ax = axes[1, 1]
    venn(ax, title="mutually exclusive (disjoint)")
    ax.add_patch(Circle((0.42, 0.5), 0.3, facecolor=C_RED, alpha=0.5, edgecolor=C_BLUE, lw=2))
    ax.add_patch(Circle((0.88, 0.5), 0.3, facecolor=C_RED, alpha=0.5, edgecolor=C_GREEN, lw=2))
    ax.text(0.24, 0.5, "A", fontsize=13, fontweight="bold", color=C_BLUE)
    ax.text(1.05, 0.5, "B", fontsize=13, fontweight="bold", color=C_GREEN)
    ax.text(0.65, 0.16, "no overlap → P(A∩B) = 0", fontsize=9, ha="center")
    fig.suptitle("Venn diagrams: the visual language of events", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "prob-venn.png")


def prob_pmf_pdf_cdf():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    # discrete PMF (binomial-ish)
    ax = axes[0, 0]
    xs = np.arange(0, 7)
    p = [0.05, 0.15, 0.25, 0.25, 0.17, 0.09, 0.04]
    ax.bar(xs, p, color=C_BLUE, width=0.6, edgecolor="black", lw=0.8)
    ax.set_title("DISCRETE: PMF gives P(X = x) — exact point probabilities", fontsize=10)
    style_ax(ax, "x", "P(X = x)")
    ax = axes[0, 1]
    ax.step(np.concatenate([[xs[0] - 0.5], xs]), np.concatenate([[0], np.cumsum(p)]), where="post", color=C_RED, lw=2.2)
    ax.plot(xs, np.cumsum(p), "o", color=C_RED, ms=5)
    ax.set_title("DISCRETE: CDF is a staircase (jumps at each value)", fontsize=10)
    style_ax(ax, "x", "F(x) = P(X ≤ x)")
    # continuous Gaussian
    ax = axes[1, 0]
    x = np.linspace(-4, 4, 300)
    pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(x, pdf, color=C_BLUE, lw=2.4)
    ax.set_title("CONTINUOUS: PDF gives DENSITY, not probability (P(X=x)=0)", fontsize=10)
    style_ax(ax, "x", "f(x)")
    ax = axes[1, 1]
    import math
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
    ax.plot(x, cdf, color=C_GREEN, lw=2.4)
    ax.set_title("CONTINUOUS: CDF is smooth (P(X ≤ x))", fontsize=10)
    style_ax(ax, "x", "F(x)")
    fig.suptitle("PMF vs PDF vs CDF — discrete vs continuous", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "prob-pmf-pdf-cdf.png")


def prob_joint_table():
    rng = np.random.default_rng(3)
    joint = np.array([[0.21, 0.25], [0.30, 0.24]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(joint, cmap="YlGnBu", vmin=0, vmax=0.35)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{joint[i, j]:.2f}", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Heavy traffic", "Light traffic"])
    ax.set_yticklabels(["Rainy", "Sunny"])
    ax.set_title("JOINT table P(X, Y)\n(P(Rainy, Heavy) = 0.30)", fontsize=10.5)
    ax.xaxis.set_ticks_position("top")
    ax = axes[1]
    px = joint.sum(axis=1)
    ax.bar([0, 1], px, color=[C_BLUE, C_GREEN], width=0.5, edgecolor="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rainy", "Sunny"])
    ax.set_title("MARGINAL P(X): sum over Y\nP(Rainy)=0.21+0.30=0.51", fontsize=10.5)
    style_ax(ax, "", "P(X)")
    ax.set_ylim(0, 0.6)
    for i, v in enumerate(px):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    fig.suptitle("Joint probabilities → sum rows/columns to get marginals", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "prob-joint-table.png")


def prob_covariance_signs():
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cases = [
        (np.array([[1.0, 0.9], [0.9, 1.0]]), "Cov > 0: both rise together", C_GREEN),
        (np.array([[1.0, -0.85], [-0.85, 1.0]]), "Cov < 0: one rises, other falls", C_RED),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "Cov ≈ 0: no linear link", C_GRAY),
    ]
    for ax, (cov, title, c) in zip(axes, cases):
        d = rng.multivariate_normal([0, 0], cov, 250)
        ax.scatter(d[:, 0], d[:, 1], s=16, alpha=0.6, color=c)
        ax.set_title(title, fontsize=10)
        style_ax(ax, "X", "Y", grid=False)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
    fig.suptitle("Covariance sign = direction of co-movement (units depend on scale!)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "prob-covariance-signs.png")


def prob_independence_correlation():
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x = rng.uniform(-3, 3, 300)
    # independent + uncorrelated
    axes[0].scatter(x, rng.normal(0, 1, 300), s=14, alpha=0.6, color=C_BLUE)
    axes[0].set_title("Independent (uncorrelated)", fontsize=10)
    # dependent but uncorrelated (parabola)
    axes[1].scatter(x, x ** 2 + rng.normal(0, 0.5, 300), s=14, alpha=0.6, color=C_ORANGE)
    axes[1].set_title("DEPENDENT but\ncorrelation ≈ 0 (parabola!)", fontsize=10)
    # correlated
    y = 0.9 * x + rng.normal(0, 0.8, 300)
    axes[2].scatter(x, y, s=14, alpha=0.6, color=C_GREEN)
    axes[2].set_title("Correlated (linear link)", fontsize=10)
    # perfectly correlated
    axes[3].scatter(x, 2 * x, s=14, alpha=0.6, color=C_RED)
    axes[3].set_title("Perfectly correlated (r = 1)", fontsize=10)
    for ax in axes:
        style_ax(ax, "X", "Y", grid=False)
        ax.set_xlim(-3.6, 3.6)
    fig.suptitle("Independence ≠ zero correlation: correlation only catches LINEAR patterns",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "prob-independence-correlation.png")


def prob_bernoulli():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([0, 1], [0.3, 0.7], width=0.4, color=[C_RED, C_GREEN], edgecolor="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["X = 0 (fail)\nP = 1−p", "X = 1 (success)\nP = p"])
    ax.set_ylabel("probability")
    ax.set_title("Bernoulli(p): one coin flip — success p, failure 1−p", fontsize=11)
    style_ax(ax, "", "P(X = x)")
    ax.set_ylim(0, 0.85)
    for x_, v in [(0, 0.3), (1, 0.7)]:
        ax.text(x_, v + 0.02, f"{v}", ha="center", fontsize=12)
    savefig(fig, "prob-bernoulli.png")


import math

def prob_binomial():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n, p = 10, 0.5
    k = np.arange(0, n + 1)
    pmf = np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in k])
    ax.bar(k, pmf, width=0.6, color=C_BLUE, edgecolor="black", lw=0.6)
    ax.set_xlabel("number of heads (successes)")
    ax.set_ylabel("probability")
    ax.set_title("Binomial(n=10, p=0.5): count of successes in 10 coin flips", fontsize=11)
    style_ax(ax, "k", "P(X = k)")
    savefig(fig, "prob-binomial.png")


def prob_poisson():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lam = 3.0
    k = np.arange(0, 13)
    pmf = np.exp(-lam) * lam ** k / np.array([math.factorial(i) for i in k])
    ax.bar(k, pmf, width=0.6, color=C_PURPLE, edgecolor="black", lw=0.6)
    ax.set_xlabel("k events")
    ax.set_ylabel("probability")
    ax.set_title(r"Poisson(λ=3): number of rare events in a fixed interval", fontsize=11)
    style_ax(ax, "k", "P(X = k)")
    savefig(fig, "prob-poisson.png")


def prob_exponential():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lam = 0.5
    x = np.linspace(0, 8, 300)
    pdf = lam * np.exp(-lam * x)
    ax.plot(x, pdf, color=C_ORANGE, lw=2.5, label=r"$f(x) = \lambda e^{-\lambda x}$")
    ax.fill_between(x, pdf, alpha=0.2, color=C_ORANGE)
    ax.set_xlabel("time until next event")
    ax.set_ylabel("density")
    ax.set_title("Exponential(λ=0.5): waiting time for the next event", fontsize=11)
    style_ax(ax, "x", "f(x)")
    ax.legend()
    savefig(fig, "prob-exponential.png")


def prob_uniform():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hlines(1 / 6, 0, 6, color=C_GREEN, lw=3)
    ax.vlines(0, 0, 1 / 6, color=C_GREEN, lw=2)
    ax.vlines(6, 0, 1 / 6, color=C_GREEN, lw=2)
    ax.fill_between([0, 6], 1 / 6, alpha=0.2, color=C_GREEN)
    ax.text(3, 0.13, r"area = 6 × $\frac{1}{6}$ = 1", fontsize=11, ha="center")
    ax.set_title("Uniform[0, 6]: every value equally likely (height = 1/(b−a))", fontsize=11)
    style_ax(ax, "x", "density")
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(0, 0.25)
    savefig(fig, "prob-uniform.png")


def prob_gaussian():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mu, sigma = 0, 1
    x = np.linspace(-5, 5, 400)
    def gauss(x, m, s):
        return (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
    for m, s, c, lbl in [(0, 0.7, C_BLUE, "σ = 0.7 (narrow)"), (0, 1, C_RED, "σ = 1 (standard)"),
                         (0, 1.8, C_GREEN, "σ = 1.8 (wide)")]:
        ax.plot(x, gauss(x, m, s), color=c, lw=2.2, label=lbl)
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.annotate(r"mean $\mu$", xy=(0, 0.62), xytext=(0.5, 0.62), fontsize=11)
    ax.set_title(r"Gaussian $N(\mu, \sigma^2)$: μ shifts it, σ widens it", fontsize=12, fontweight="bold")
    style_ax(ax, "x", "density")
    ax.legend(fontsize=9)
    savefig(fig, "prob-gaussian.png")


def prob_gaussian_689599():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 500)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)
    ax.plot(x, pdf, color=C_BLUE, lw=2.5)
    for k, c, lbl in [(1, "#1f77b4", "68%"), (2, "#2ca02c", "95%"), (3, "#d62728", "99.7%")]:
        xs = np.linspace(-k, k, 100)
        ax.fill_between(xs, (1 / np.sqrt(2 * np.pi)) * np.exp(-xs ** 2 / 2), alpha=0.25, color=c)
        ax.annotate(lbl, xy=(0, 0.05), xytext=(k - 0.35, 0.09), fontsize=10, color=c)
    ax.set_title("Empirical rule: 68–95–99.7% of data within ±1σ, ±2σ, ±3σ", fontsize=12, fontweight="bold")
    style_ax(ax, "x", "density")
    savefig(fig, "prob-gaussian-689599.png")


def prob_beta():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0.001, 0.999, 400)
    def beta_pdf(x, a, b):
        B = (math.gamma(a) * math.gamma(b)) / math.gamma(a + b)
        return x ** (a - 1) * (1 - x) ** (b - 1) / B
    for (a, b), c, lbl in [((0.5, 0.5), C_ORANGE, "Beta(0.5, 0.5)"), ((2, 5), C_BLUE, "Beta(2, 5)"),
                           ((5, 2), C_GREEN, "Beta(5, 2)"), ((2, 2), C_PURPLE, "Beta(2, 2)")]:
        ax.plot(x, beta_pdf(x, a, b), color=c, lw=2.2, label=lbl)
    ax.set_title("Beta distribution: a distribution OVER a probability p (between 0 and 1)",
                 fontsize=11.5, fontweight="bold")
    style_ax(ax, "p (probability)", "density")
    ax.legend(fontsize=9)
    savefig(fig, "prob-beta.png")


def prob_softmax():
    logits = np.array([2.0, 1.0, 0.1])
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ["dog", "cat", "bird"]
    ax = axes[0]
    ax.bar(labels, logits, color=C_BLUE, width=0.5, edgecolor="black")
    ax.set_title("raw logits (any real numbers)", fontsize=11)
    style_ax(ax, "", "logit score")
    ax = axes[1]
    ax.bar(labels, probs, color=C_GREEN, width=0.5, edgecolor="black")
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_title("softmax output (all positive, sum = 1)", fontsize=11)
    style_ax(ax, "", "probability")
    ax.set_ylim(0, 1)
    fig.suptitle("Softmax: turning arbitrary scores into a valid probability distribution",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "prob-softmax.png")


def prob_entropy():
    fig, ax = plt.subplots(figsize=(7, 5))
    p = np.linspace(0.001, 0.999, 300)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    ax.plot(p, H, color=C_BLUE, lw=2.5)
    ax.axvline(0.5, color=C_GRAY, lw=1, ls=":")
    ax.axhline(1.0, color=C_GRAY, lw=1, ls=":")
    ax.scatter([0.5], [1.0], color=C_RED, s=60, zorder=6)
    ax.annotate("max entropy at p = 0.5 (complete uncertainty)", xy=(0.5, 1.0),
                xytext=(0.56, 0.86), fontsize=10, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.annotate("p = 0.1 → very predictable,\nlow entropy", xy=(0.1, 0.47), xytext=(0.14, 0.6),
                fontsize=10, color=C_GREEN, arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.set_title("Binary entropy H(p): uncertainty vs. probability of success", fontsize=12, fontweight="bold")
    style_ax(ax, "p", "H(p) (bits)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    savefig(fig, "prob-entropy.png")


def prob_cross_entropy():
    P = np.array([0.6, 0.2, 0.15, 0.05])
    Q = np.array([0.4, 0.3, 0.25, 0.05])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    xpos = np.arange(4)
    ax.bar(xpos - 0.18, P, width=0.36, color=C_BLUE, edgecolor="black", label="true P")
    ax.bar(xpos + 0.18, Q, width=0.36, color=C_ORANGE, edgecolor="black", label="predicted Q")
    ax.set_xticks(xpos); ax.set_xticklabels(["cat", "dog", "bird", "fish"])
    ax.set_title("P vs Q: cross-entropy is larger when Q is wrong where P is big", fontsize=10)
    style_ax(ax, "", "probability")
    ax.legend(fontsize=9)
    ax = axes[1]
    terms = -P * np.log2(Q)
    ax.bar(xpos, terms, color=C_GREEN, edgecolor="black", width=0.55)
    ax.set_xticks(xpos); ax.set_xticklabels(["cat", "dog", "bird", "fish"])
    for i, v in enumerate(terms):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_title(r"per-class $−P_i \log_2 Q_i$  (sum = cross-entropy)", fontsize=10)
    style_ax(ax, "", "surprise contribution")
    fig.suptitle(f"Cross-entropy: H(P,Q) = Σ −P·log₂(Q) = {(-P * np.log2(Q)).sum():.2f} bits",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "prob-cross-entropy.png")


def prob_monte_carlo_pi():
    rng = np.random.default_rng(1)
    N = 4000
    pts = rng.uniform(-1, 1, (N, 2))
    inside = np.linalg.norm(pts, axis=1) <= 1
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(pts[~inside, 0], pts[~inside, 1], s=4, color=C_RED, alpha=0.5, label="outside circle")
    ax.scatter(pts[inside, 0], pts[inside, 1], s=4, color=C_BLUE, alpha=0.5, label="inside circle")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=C_GREEN, lw=2.5)
    est = 4 * inside.mean()
    ax.set_title(f"Monte Carlo π-estimate: 4 × {inside.mean():.3f} = {est:.3f}", fontsize=12, fontweight="bold")
    style_ax(ax, "x", "y", grid=False)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    savefig(fig, "prob-montecarlo-pi.png")


def prob_monte_carlo_convergence():
    rng = np.random.default_rng(2)
    N = 10000
    pts = rng.uniform(-1, 1, (N, 2))
    inside = np.cumsum(np.linalg.norm(pts, axis=1) <= 1)
    est = 4 * inside / np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, N + 1), est, color=C_BLUE, lw=1.4)
    ax.axhline(np.pi, color=C_RED, lw=2, ls="--", label="true π ≈ 3.14159")
    ax.set_xscale("log")
    ax.set_title("Estimate improves with more samples (law of large numbers in action)", fontsize=11.5)
    style_ax(ax, "number of samples N", "estimate of π")
    ax.legend()
    ax.set_ylim(2.6, 3.7)
    savefig(fig, "prob-montecarlo-convergence.png")


def prob_lln():
    rng = np.random.default_rng(4)
    rolls = rng.integers(1, 7, 5000)
    means = np.cumsum(rolls) / np.arange(1, 5001)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, 5001), means, color=C_BLUE, lw=1.4, label="running average")
    ax.axhline(3.5, color=C_RED, lw=2, ls="--", label="true expected value E[X] = 3.5")
    ax.set_xscale("log")
    ax.set_title("Law of Large Numbers: sample mean → true mean as n grows", fontsize=11.5)
    style_ax(ax, "number of rolls n", "sample mean")
    ax.legend()
    ax.set_ylim(3.0, 4.0)
    savefig(fig, "prob-lln.png")


def prob_dirichlet():
    fig, ax = plt.subplots(figsize=(7, 6.2))
    rng = np.random.default_rng(6)
    alpha = np.array([2, 2, 2])
    n = 2500
    draws = rng.dirichlet(alpha, n)
    # map 3-simplex to 2D triangle (barycentric → cartesian)
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    pts2d = draws @ tri
    ax.scatter(pts2d[:, 0], pts2d[:, 1], s=5, alpha=0.4, color=C_BLUE)
    ax.add_patch(Polygon(tri, closed=True, fill=False, edgecolor=C_RED, lw=2.5))
    ax.text(0.5, 0.06, "component 1", ha="center", fontsize=9)
    ax.text(0.06, 0.5, "component 2", rotation=60, fontsize=9, va="center")
    ax.text(0.94, 0.5, "component 3", rotation=-60, fontsize=9, va="center")
    ax.set_title(r"Dirichlet(2,2,2): points = probability vectors over 3 classes (each sums to 1)",
                 fontsize=10.5, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    savefig(fig, "prob-dirichlet.png")


# ======================================================================
# STATISTICS
# ======================================================================

def stat_histogram():
    rng = np.random.default_rng(8)
    data = rng.normal(170, 8, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(data, bins=25, color=C_BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(data), color=C_RED, lw=2.5, label=f"mean = {np.mean(data):.1f}")
    ax.axvline(np.median(data), color=C_GREEN, lw=2.5, ls="--", label=f"median = {np.median(data):.1f}")
    ax.set_title("Histogram: how often each value range occurs (here: heights in cm)", fontsize=11.5)
    style_ax(ax, "height (cm)", "count")
    ax.legend(fontsize=9)
    savefig(fig, "stat-histogram.png")


def stat_mean_median_mode():
    rng = np.random.default_rng(9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    # symmetric
    d = rng.normal(0, 1, 5000)
    ax = axes[0]
    ax.hist(d, bins=40, color=C_BLUE, alpha=0.7)
    ax.axvline(np.mean(d), color=C_RED, lw=2.5, label="mean")
    ax.axvline(np.median(d), color=C_GREEN, lw=2.5, ls="--", label="median")
    ax.set_title("Symmetric: mean ≈ median", fontsize=10.5)
    style_ax(ax, "", "count", grid=False)
    ax.legend(fontsize=9)
    # right-skewed
    d = np.concatenate([rng.normal(0, 1, 4500), rng.exponential(3, 500)])
    ax = axes[1]
    ax.hist(d, bins=50, color=C_ORANGE, alpha=0.7)
    ax.axvline(np.mean(d), color=C_RED, lw=2.5, label="mean (pulled right)")
    ax.axvline(np.median(d), color=C_GREEN, lw=2.5, ls="--", label="median (robust)")
    ax.set_title("Right-skewed (long tail): mean > median", fontsize=10.5)
    style_ax(ax, "", "count", grid=False)
    ax.legend(fontsize=9)
    # left-skewed
    d = np.concatenate([rng.normal(0, 1, 4500), -rng.exponential(3, 500)])
    ax = axes[2]
    ax.hist(d, bins=50, color=C_PURPLE, alpha=0.7)
    ax.axvline(np.mean(d), color=C_RED, lw=2.5, label="mean (pulled left)")
    ax.axvline(np.median(d), color=C_GREEN, lw=2.5, ls="--", label="median")
    ax.set_title("Left-skewed (long left tail): mean < median", fontsize=10.5)
    style_ax(ax, "", "count", grid=False)
    ax.legend(fontsize=9)
    fig.suptitle("Skewness pulls the MEAN toward the tail; the MEDIAN stays central", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-mean-median-mode.png")


def stat_variance():
    rng = np.random.default_rng(10)
    mu = 5.0
    low = np.array([4.5, 5.5, 4.8, 5.2, 5.0, 4.6, 5.4, 4.9, 5.1, 5.3])
    high = np.array([0.0, 10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, data, title, c in [(axes[0], low, f"low variance: all close to μ={mu}", C_GREEN),
                               (axes[1], high, f"high variance: spread far from μ={mu}", C_RED)]:
        ax.axvline(mu, color="black", lw=1.2, ls="--", label="mean μ")
        for v in data:
            ax.plot([v, v], [0, 1], color=c, alpha=0.4, lw=0.8)
            ax.plot([v], [0.5], "o", color=c, ms=6)
        ax.set_title(title, fontsize=10.5)
        style_ax(ax, "value", "", grid=False)
        ax.set_ylim(-0.1, 1.2)
        ax.set_yticks([])
        ax.legend(fontsize=9)
    fig.suptitle("Variance = average squared distance of points from the mean", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-variance.png")


def stat_boxplot():
    rng = np.random.default_rng(12)
    data = np.concatenate([rng.normal(50, 10, 90), [120]])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, vert=False, widths=0.4, patch_artist=True)
    bp["boxes"][0].set_facecolor(C_LIGHT)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    ax.scatter([120], [1], color=C_RED, s=60, zorder=6,
               label="outlier (> Q3 + 1.5×IQR)")
    ax.axvline(q1 - 1.5 * iqr, color=C_GRAY, lw=1, ls=":")
    ax.axvline(q3 + 1.5 * iqr, color=C_GRAY, lw=1, ls=":")
    ax.text(q1, 1.45, "Q1", fontsize=11, ha="center", color=C_BLUE)
    ax.text(med, 1.45, "median", fontsize=11, ha="center", color=C_GREEN)
    ax.text(q3, 1.45, "Q3", fontsize=11, ha="center", color=C_ORANGE)
    ax.annotate("", xy=(q3, 0.98), xytext=(q1, 0.98),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.6))
    ax.text((q1 + q3) / 2, 1.18, "IQR (middle 50%)", fontsize=10, ha="center", color=C_RED)
    ax.set_title("Box plot: min, Q1, median, Q3, max + the IQR outlier rule", fontsize=11.5)
    style_ax(ax, "value", "", grid=False)
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 130)
    savefig(fig, "stat-boxplot.png")


def stat_standardization():
    rng = np.random.default_rng(13)
    raw = rng.normal(65, 12, 500)
    z = (raw - np.mean(raw)) / np.std(raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(raw, bins=30, color=C_BLUE, alpha=0.75)
    axes[0].set_title(f"raw: mean = {np.mean(raw):.1f}, σ = {np.std(raw):.1f}", fontsize=10.5)
    style_ax(axes[0], "original units", "count", grid=False)
    axes[1].hist(z, bins=30, color=C_GREEN, alpha=0.75)
    axes[1].axvline(0, color=C_RED, lw=2, ls="--")
    axes[1].set_title("z-scored: mean = 0, σ = 1 (unitless)", fontsize=10.5)
    style_ax(axes[1], "z = (x − μ)/σ", "count", grid=False)
    fig.suptitle("Standardization: every feature becomes comparable (same scale)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-standardization.png")


def stat_correlation_scatter():
    rng = np.random.default_rng(14)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    cases = [
        (0.95, "r ≈ +0.95: strong positive", C_GREEN),
        (0.3, "r ≈ +0.3: weak positive", C_BLUE),
        (0.0, "r ≈ 0: no linear relation", C_GRAY),
        (-0.85, "r ≈ −0.85: strong negative", C_RED),
    ]
    for ax, (r, title, c) in zip(axes.ravel(), cases):
        x = rng.normal(0, 1, 300)
        y = r * x + np.sqrt(1 - r ** 2) * rng.normal(0, 1, 300)
        ax.scatter(x, y, s=14, alpha=0.6, color=c)
        ax.set_title(title, fontsize=10.5)
        style_ax(ax, "X", "Y", grid=False)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
    fig.suptitle("Pearson correlation r: strength + direction of the LINEAR relationship",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "stat-correlation-scatter.png")


def stat_clt():
    rng = np.random.default_rng(15)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    # population: exponential (very non-normal)
    pop = rng.exponential(1.0, 100000)
    axes[0].hist(pop, bins=60, color=C_ORANGE, alpha=0.8)
    axes[0].set_title("1) POPULATION (exponential — not normal!)", fontsize=10)
    style_ax(axes[0], "value", "density", grid=False)
    axes[1].hist(rng.choice(pop, 30), bins=15, color=C_BLUE, alpha=0.8)
    axes[1].set_title("2) ONE random sample (n = 30)", fontsize=10)
    style_ax(axes[1], "value", "count", grid=False)
    means = [np.mean(rng.choice(pop, 30)) for _ in range(2000)]
    axes[2].hist(means, bins=50, color=C_GREEN, alpha=0.8)
    axes[2].set_title("3) one sample MEAN", fontsize=10)
    style_ax(axes[2], "mean", "count", grid=False)
    many = np.array([np.mean(rng.choice(pop, 30)) for _ in range(10000)])
    axes[3].hist(many, bins=60, color=C_PURPLE, alpha=0.8, density=True)
    xs = np.linspace(many.min(), many.max(), 200)
    axes[3].plot(xs, (1 / (np.std(many) * np.sqrt(2 * np.pi))) *
                 np.exp(-0.5 * ((xs - np.mean(many)) / np.std(many)) ** 2),
                 color=C_RED, lw=2.2, label="normal fit")
    axes[3].set_title("4) distribution of 10,000 sample means → NORMAL!", fontsize=10)
    style_ax(axes[3], "sample mean", "density", grid=False)
    axes[3].legend(fontsize=8)
    fig.suptitle("Central Limit Theorem: means of any distribution become Gaussian", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-clt.png")


def stat_se_vs_n():
    sigma = 2.0
    n = np.arange(5, 300)
    se = sigma / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(n, se, color=C_BLUE, lw=2.5)
    ax.set_xlabel("sample size n"); ax.set_ylabel("standard error")
    ax.set_title(r"Standard error SE = σ/√n shrinks as n grows (4× the data → ½ the SE)",
                 fontsize=11)
    style_ax(ax, "n", "SE")
    savefig(fig, "stat-se-vs-n.png")


def stat_confidence_interval():
    rng = np.random.default_rng(16)
    mu, sigma, n = 100, 15, 30
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    x = np.linspace(mu - 4 * sigma / np.sqrt(n), mu + 4 * sigma / np.sqrt(n), 300)
    se = sigma / np.sqrt(n)
    pdf = (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / se) ** 2)
    ax.plot(x, pdf, color=C_BLUE, lw=2.5)
    xs = np.linspace(mu - 1.96 * se, mu + 1.96 * se, 100)
    ax.fill_between(xs, (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / se) ** 2),
                    color=C_GREEN, alpha=0.5)
    ax.axvline(mu, color=C_RED, lw=2, ls="--", label="true population mean μ")
    ax.text(mu, 0.75 * pdf.max(), "middle 95%", ha="center", fontsize=11, color=C_GREEN)
    ax.set_title("Sampling distribution of the sample mean", fontsize=11)
    style_ax(ax, "sample mean x̄", "density", grid=False)
    ax.legend(fontsize=9)
    ax = axes[1]
    rng2 = np.random.default_rng(17)
    for i in range(30):
        sample = rng2.normal(mu, sigma, n)
        m = np.mean(sample)
        err = 1.96 * sigma / np.sqrt(n)
        c = C_GREEN if (mu - err <= m <= mu + err) else C_RED
        ax.plot([m - err, m + err], [i, i], color=c, lw=2.5)
        ax.plot([m], [i], "o", color=c, ms=3)
    ax.axvline(mu, color="black", lw=2, ls="--")
    ax.set_title("30 different samples: ~95% of CIs contain μ (red = missed)", fontsize=11)
    style_ax(ax, "x̄", "sample #", grid=False)
    fig.suptitle("Confidence interval: if we repeat sampling, ~95% of intervals contain the truth",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-confidence-interval.png")


def stat_pvalue():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-4, 4, 400)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)
    ax.plot(x, pdf, color=C_BLUE, lw=2.5, label="null distribution (H₀ true)")
    obs = 2.1
    xs = np.linspace(obs, 4, 100)
    ax.fill_between(xs, (1 / np.sqrt(2 * np.pi)) * np.exp(-xs ** 2 / 2), color=C_RED, alpha=0.55,
                    label="p-value = tail area")
    ax.axvline(obs, color=C_RED, lw=2, ls="--")
    ax.annotate("observed statistic = 2.1", xy=(obs, 0.02), xytext=(obs + 0.3, 0.15),
                fontsize=10, color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    import math
    p = math.erfc(obs / np.sqrt(2)) / 2
    ax.text(2.6, 0.3, f"p ≈ {p:.4f}", fontsize=12, color=C_RED)
    ax.set_title("p-value = probability of getting data this extreme IF H₀ were true", fontsize=11)
    style_ax(ax, "test statistic", "density", grid=False)
    ax.legend(fontsize=9)
    savefig(fig, "stat-pvalue.png")


def stat_type1_type2():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(-4, 6, 500)
    h0 = (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)
    ha = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 2.5) / 1) ** 2)
    ax.plot(x, h0, color=C_BLUE, lw=2.5, label="H₀: no effect (mean 0)")
    ax.plot(x, ha, color=C_GREEN, lw=2.5, label="Hₐ: effect exists (mean 2.5)")
    crit = 1.96
    xs = np.linspace(crit, 4, 100)
    ax.fill_between(xs, (1 / np.sqrt(2 * np.pi)) * np.exp(-xs ** 2 / 2), color=C_RED, alpha=0.6)
    ax.text(2.9, 0.16, "Type I (α)\nfalse positive", fontsize=9, color=C_RED, ha="center")
    xs = np.linspace(-4, crit, 300)
    ax.fill_between(xs, (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - 2.5) / 1) ** 2),
                    color=C_ORANGE, alpha=0.6)
    ax.text(0.6, 0.28, "Type II (β)\nfalse negative", fontsize=9, color="#b85c00", ha="center")
    ax.axvline(crit, color="black", lw=2, ls="--", label="decision boundary (α = 0.05)")
    ax.set_title("Type I vs Type II errors: overlapping distributions make mistakes unavoidable",
                 fontsize=11)
    style_ax(ax, "test statistic", "density", grid=False)
    ax.legend(fontsize=8.5, loc="upper left")
    savefig(fig, "stat-type1-type2.png")


def stat_mle_bernoulli():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    p = np.linspace(0.001, 0.999, 300)
    heads, flips = 7, 10
    like = p ** heads * (1 - p) ** (flips - heads)
    ax.plot(p, like, color=C_BLUE, lw=2.5, label="likelihood L(p)")
    mle = heads / flips
    ax.axvline(mle, color=C_RED, lw=2, ls="--", label=f"MLE p̂ = {mle}")
    ax.scatter([mle], [mle ** heads * (1 - mle) ** (flips - heads)], color=C_RED, s=70, zorder=6)
    ax.set_title("MLE for a coin: which p makes 7 heads out of 10 MOST likely?", fontsize=11.5)
    style_ax(ax, "p", "L(p)")
    ax.legend(fontsize=9)
    savefig(fig, "stat-mle-bernoulli.png")


def stat_map():
    fig, ax = plt.subplots(figsize=(8, 5))
    p = np.linspace(0.001, 0.999, 300)
    a_prior, b_prior = 2, 2
    heads, flips = 7, 10
    prior = p ** (a_prior - 1) * (1 - p) ** (b_prior - 1)
    like = p ** heads * (1 - p) ** (flips - heads)
    post = p ** (a_prior + heads - 1) * (1 - p) ** (b_prior + flips - heads - 1)
    for arr, c, lbl in [(prior, C_ORANGE, "prior Beta(2,2)"),
                        (like / like.max(), C_BLUE, "likelihood (7 heads / 10)"),
                        (post / post.max(), C_GREEN, "posterior (shifts toward data)")]:
        ax.plot(p, arr, color=c, lw=2.2, label=lbl)
    map_p = (a_prior + heads - 1) / (a_prior + b_prior + flips - 2)
    ax.axvline(map_p, color=C_GREEN, lw=2, ls="--")
    ax.annotate(f"MAP = {map_p:.2f}", xy=(map_p, 0.92), xytext=(map_p + 0.08, 0.8),
                fontsize=10, color=C_GREEN, arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.set_title("MAP: posterior ∝ prior × likelihood (belief updated by evidence)", fontsize=11.5)
    style_ax(ax, "p", "density (normalized)")
    ax.legend(fontsize=9, loc="upper right")
    savefig(fig, "stat-map.png")


def stat_bootstrap():
    rng = np.random.default_rng(18)
    orig = rng.normal(50, 8, 100)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].hist(orig, bins=15, color=C_BLUE, alpha=0.8)
    axes[0].set_title("original sample (n=100)", fontsize=10)
    style_ax(axes[0], "", "count", grid=False)
    for i, c in [(1, C_GREEN), (2, C_ORANGE), (3, C_PURPLE)]:
        resample = rng.choice(orig, 100, replace=True)
        axes[i].hist(resample, bins=15, color=c, alpha=0.8)
        axes[i].set_title(f"bootstrap resample #{i} (with replacement)", fontsize=10)
        style_ax(axes[i], "", "count", grid=False)
    fig.suptitle("Bootstrapping: repeatedly resample WITH replacement to simulate new datasets",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(fig, "stat-bootstrap.png")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("Generating CALCULUS images...")
    calc_slope_line(); calc_secant_tangent(); calc_tangent(); calc_limit()
    calc_partial_derivative(); calc_gradient_contour(); calc_concavity()
    calc_jacobian(); calc_hessian_surfaces(); calc_convex_nonconvex()
    calc_gd_1d(); calc_gd_2d(); calc_learning_rate(); calc_momentum()
    calc_riemann(); calc_definite_integral(); calc_gaussian_pdf(); calc_diffusion_noise()

    print("Generating LINEAR ALGEBRA images...")
    linalg_vector_basics(); linalg_vector_add(); linalg_vector_scale()
    linalg_dot_product(); linalg_norms(); linalg_unit_circles()
    linalg_cosine_similarity(); linalg_cross_product(); linalg_matrix_vector()
    linalg_matmul(); linalg_transpose(); linalg_det_area()
    linalg_transformations(); linalg_rank(); linalg_span()
    linalg_linear_system(); linalg_eigenvectors(); linalg_pca()
    linalg_svd(); linalg_tensors()

    print("Generating PROBABILITY images...")
    prob_venn(); prob_pmf_pdf_cdf(); prob_joint_table()
    prob_covariance_signs(); prob_independence_correlation(); prob_bernoulli()
    prob_binomial(); prob_poisson(); prob_exponential(); prob_uniform()
    prob_gaussian(); prob_gaussian_689599(); prob_beta(); prob_softmax()
    prob_entropy(); prob_cross_entropy(); prob_monte_carlo_pi()
    prob_monte_carlo_convergence(); prob_lln(); prob_dirichlet()

    print("Generating STATISTICS images...")
    stat_histogram(); stat_mean_median_mode(); stat_variance()
    stat_boxplot(); stat_standardization(); stat_correlation_scatter()
    stat_clt(); stat_se_vs_n(); stat_confidence_interval(); stat_pvalue()
    stat_type1_type2(); stat_mle_bernoulli(); stat_map(); stat_bootstrap()

    print(f"\nAll done. {len(os.listdir(OUT))} images in {OUT}")


if __name__ == "__main__":
    main()
