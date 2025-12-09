"""
Visualization functions for optimization problems.
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.patches import Circle

from .optimization import feasible


def plot(P, resolution=200, palette=None, numeric="Real"):
    """
    Create a scatter plot visualization of the optimization problem.
    
    Parameters
    ----------
    P : Optimization
        Optimization problem
    resolution : int, optional
        Grid resolution (default 200)
    palette : str, optional
        Matplotlib colormap (default 'gray')
    numeric : str, optional
        Numeric type (default 'Real')
    """
    dot_alpha = 1
    if palette is None:
        palette = "gray"
        dot_alpha = 0.7

    xs = P.symbols()
    d1, d2 = P.variables
    xmin, xmax = 0, d1.upper
    ymin, ymax = 0, d2.upper

    if d1.numeric == "Real":
        x_vals = np.linspace(xmin, xmax, resolution)
    elif d1.numeric == "Integer": 
        x_vals = np. arange(int(xmin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    if d2.numeric == "Real": 
        y_vals = np. linspace(ymin, ymax, resolution)
    elif d2.numeric == "Integer":
        y_vals = np.arange(int(ymin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    X, Y = np.meshgrid(x_vals, y_vals)

    if P.objective is None:
        raise ValueError("No objective function defined.")

    f_num = sym.lambdify(xs, P.objective, "numpy")
    Z = f_num(X, Y)

    masks = []
    for r in P.restrictions:
        g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
        G = g_num(X, Y)
        masks.append(G <= 0)

    feasible_mask = np.ones_like(X, dtype=bool)
    for mask in masks:
        feasible_mask &= mask

    fig, ax = plt.subplots(figsize=(8, 8))

    colors_rgb = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    colors_hsv = rgb_to_hsv(colors_rgb)

    shape = X.shape
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)

    for i, mask in enumerate(masks):
        h, s, v = colors_hsv[i % len(colors_hsv)]
        angle = 2 * np.pi * h

        cos_h = np.cos(angle)
        sin_h = np.sin(angle)

        sum_x[mask] += cos_h
        sum_y[mask] += sin_h
        count[mask] += 1.0

    h_bg = np.zeros(shape, dtype=float)
    s_bg = np.zeros(shape, dtype=float)
    v_bg = np.zeros(shape, dtype=float)

    active = count > 0

    angles = np.arctan2(sum_y[active], sum_x[active])
    h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)

    s_bg[active] = 1.0

    v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

    s_bg[feasible_mask] = 0.0
    v_bg[feasible_mask] = 1.0

    hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
    rgb_bg = hsv_to_rgb(hsv_bg)

    ax.imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )

    satisfied_count = sum(mask for mask in masks)
    violated = len(masks) - satisfied_count

    n_points = resolution**2
    base_r = 6e4 / n_points
    R = base_r * satisfied_count / (1 + violated)

    ax.scatter(X. flatten(), Y.flatten(),
            s=R. flatten(),
            c=Z.flatten(),
            cmap=palette,
            edgecolors="none",
            alpha=dot_alpha)

    Z_feasible = np.where(feasible_mask, Z, np.nan)
    if P.kind == "max":
        opt = np.nanargmax(Z_feasible)
    else:
        opt = np.nanargmin(Z_feasible)
    x_opt = X. flatten()[opt]
    y_opt = Y.flatten()[opt]
    f_opt = Z. flatten()[opt]

    ax.scatter([x_opt], [y_opt],
            color="gold",
            edgecolor="black",
            s=220, zorder=5)

    opt_mask = (Z_feasible == f_opt)
    x_opt_all = X[opt_mask]
    y_opt_all = Y[opt_mask]

    ax.set_title(f"{P.kind}:  ≈ f({x_opt:. 2f}, {y_opt:.2f}) = {f_opt:.2f}")
    ax.set_xlabel(d1.name)
    ax.set_ylabel(d2.name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.scatter(
        x_opt_all, y_opt_all,
        color="gold",
        edgecolor="goldenrod",
        s=40,
        zorder=5
    )

    plt.show()


def field(P, resolution=200, palette=None, numeric="Real"):
    """Create a vector field visualization of the optimization problem."""
    dot_alpha = 1
    if palette is None:
        palette = "gray"
        dot_alpha = 0.7

    xs = P.symbols()
    d1, d2 = P. variables
    xmin, xmax = 0, d1.upper
    ymin, ymax = 0, d2.upper

    if d1.numeric == "Real":
        x_vals = np.linspace(xmin, xmax, resolution)
    elif d1.numeric == "Integer": 
        x_vals = np. arange(int(xmin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    if d2.numeric == "Real":
        y_vals = np.linspace(ymin, ymax, resolution)
    elif d2.numeric == "Integer":
        y_vals = np.arange(int(ymin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    X, Y = np.meshgrid(x_vals, y_vals)

    if P.objective is None:
        raise ValueError("No objective function defined.")

    f_num = sym. lambdify(xs, P. objective, "numpy")
    Z = f_num(X, Y)

    masks = []
    for r in P.restrictions:
        g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
        G = g_num(X, Y)
        masks.append(G <= 0)

    feasible_mask = np.ones_like(X, dtype=bool)
    for mask in masks:
        feasible_mask &= mask

    fig, ax = plt.subplots(figsize=(8, 8))

    colors_rgb = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    colors_hsv = rgb_to_hsv(colors_rgb)

    shape = X.shape
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)

    for i, mask in enumerate(masks):
        h, s, v = colors_hsv[i % len(colors_hsv)]
        angle = 2 * np.pi * h
        cos_h = np.cos(angle)
        sin_h = np.sin(angle)
        sum_x[mask] += cos_h
        sum_y[mask] += sin_h
        count[mask] += 1.0

    h_bg = np.zeros(shape, dtype=float)
    s_bg = np.zeros(shape, dtype=float)
    v_bg = np.zeros(shape, dtype=float)

    active = count > 0
    angles = np.arctan2(sum_y[active], sum_x[active])
    h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)
    s_bg[active] = 1.0
    v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

    s_bg[feasible_mask] = 0.0
    v_bg[feasible_mask] = 1.0

    hsv_bg = np. stack([h_bg, s_bg, v_bg], axis=-1)
    rgb_bg = hsv_to_rgb(hsv_bg)

    ax.imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )

    n_constraints = len(masks)
    n_points = X.size

    if n_constraints > 0:
        satisfied_count = np.zeros_like(X, dtype=float)
        for mask in masks:
            satisfied_count += mask
        ratio = satisfied_count / n_constraints
    else:
        ratio = np.ones_like(X, dtype=float)

    size_factor = 0.3 + 0.7 * ratio

    grad = [sym.diff(P.objective, var) for var in xs]
    gx_num = sym.lambdify(xs, grad[0], "numpy")
    gy_num = sym.lambdify(xs, grad[1], "numpy")

    GX_scalar = gx_num(X, Y)
    GY_scalar = gy_num(X, Y)

    if np.isscalar(GX_scalar):
        GX = np.full_like(X, GX_scalar)
    else:
        GX = GX_scalar

    if np. isscalar(GY_scalar):
        GY = np.full_like(Y, GY_scalar)
    else:
        GY = GY_scalar

    norm = np.hypot(GX, GY)
    norm[norm == 0] = 1.0

    base_len = 0.03 * max(xmax - xmin, ymax - ymin)

    U = GX / norm * base_len * size_factor
    V = GY / norm * base_len * size_factor

    step = max(1, resolution // 25)

    ax.quiver(
        X[:: step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        Z[::step, ::step],
        cmap=palette,
        alpha=dot_alpha,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004
    )

    Z_feasible = np.where(feasible_mask, Z, np.nan)
    if P.kind == "max":
        opt = np.nanargmax(Z_feasible)
    else:
        opt = np.nanargmin(Z_feasible)

    x_opt = X.flatten()[opt]
    y_opt = Y.flatten()[opt]
    f_opt = Z.flatten()[opt]

    ax.scatter(
        [x_opt], [y_opt],
        color="gold",
        edgecolor="black",
        s=220,
        zorder=5
    )

    opt_mask = (Z_feasible == f_opt)
    x_opt_all = X[opt_mask]
    y_opt_all = Y[opt_mask]

    ax.set_title(f"{P.kind}: ≈ f({x_opt:.2f}, {y_opt:.2f}) = {f_opt:.2f}")
    ax.set_xlabel(d1.name)
    ax.set_ylabel(d2.name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    ax.scatter(
        x_opt_all, y_opt_all,
        color="gold",
        edgecolor="goldenrod",
        s=40,
        zorder=5
    )

    plt.show()


def field_contour(P, resolution=200, palette=None, numeric="Real"):
    """Create a contour plot with vector field visualization."""
    dot_alpha = 1
    if palette is None:
        palette = "gray"
        dot_alpha = 0.7

    xs = P.symbols()
    d1, d2 = P.variables
    xmin, xmax = 0, d1.upper
    ymin, ymax = 0, d2.upper

    if d1.numeric == "Real": 
        x_vals = np. linspace(xmin, xmax, resolution)
    elif d1.numeric == "Integer":
        x_vals = np.arange(int(xmin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    if d2.numeric == "Real":
        y_vals = np.linspace(ymin, ymax, resolution)
    elif d2.numeric == "Integer": 
        y_vals = np.arange(int(ymin), int(xmax) + 1)
    else:
        raise ValueError("Numeric type not supported")

    X, Y = np.meshgrid(x_vals, y_vals)

    if P.objective is None:
        raise ValueError("No objective function defined.")

    f_num = sym.lambdify(xs, P.objective, "numpy")
    Z = f_num(X, Y)

    masks = []
    for r in P.restrictions:
        g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
        G = g_num(X, Y)
        masks.append(G <= 0)

    feasible_mask = np. ones_like(X, dtype=bool)
    for mask in masks:
        feasible_mask &= mask

    fig, ax = plt.subplots(figsize=(8, 8))

    colors_rgb = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    colors_hsv = rgb_to_hsv(colors_rgb)

    shape = X.shape
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)

    for i, mask in enumerate(masks):
        h, s, v = colors_hsv[i % len(colors_hsv)]
        angle = 2 * np.pi * h
        cos_h = np.cos(angle)
        sin_h = np.sin(angle)
        sum_x[mask] += cos_h
        sum_y[mask] += sin_h
        count[mask] += 1.0

    h_bg = np.zeros(shape, dtype=float)
    s_bg = np.zeros(shape, dtype=float)
    v_bg = np.zeros(shape, dtype=float)

    active = count > 0
    angles = np.arctan2(sum_y[active], sum_x[active])
    h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)
    s_bg[active] = 1.0
    v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

    s_bg[feasible_mask] = 0.0
    v_bg[feasible_mask] = 1.0

    hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
    rgb_bg = hsv_to_rgb(hsv_bg)

    ax.imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )

    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.5, linewidths=0.8)

    n_constraints = len(masks)
    n_points = X.size

    if n_constraints > 0:
        satisfied_count = np.zeros_like(X, dtype=float)
        for mask in masks:
            satisfied_count += mask
        ratio = satisfied_count / n_constraints
    else:
        ratio = np.ones_like(X, dtype=float)

    size_factor = 0.3 + 0.7 * ratio

    grad = [sym.diff(P.objective, var) for var in xs]
    gx_num = sym.lambdify(xs, grad[0], "numpy")
    gy_num = sym.lambdify(xs, grad[1], "numpy")

    GX_scalar = gx_num(X, Y)
    GY_scalar = gy_num(X, Y)

    if np.isscalar(GX_scalar):
        GX = np.full_like(X, GX_scalar)
    else:
        GX = GX_scalar

    if np.isscalar(GY_scalar):
        GY = np.full_like(Y, GY_scalar)
    else:
        GY = GY_scalar

    norm = np.hypot(GX, GY)
    norm[norm == 0] = 1.0

    base_len = 0.03 * max(xmax - xmin, ymax - ymin)

    U = GX / norm * base_len * size_factor
    V = GY / norm * base_len * size_factor

    step = max(1, resolution // 25)

    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, :: step],
        V[::step, ::step],
        Z[::step, ::step],
        cmap=palette,
        alpha=dot_alpha,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004
    )

    Z_feasible = np.where(feasible_mask, Z, np. nan)
    if P.kind == "max":
        opt = np.nanargmax(Z_feasible)
    else:
        opt = np.nanargmin(Z_feasible)

    x_opt = X. flatten()[opt]
    y_opt = Y.flatten()[opt]
    f_opt = Z. flatten()[opt]

    ax.scatter(
        [x_opt], [y_opt],
        color="gold",
        edgecolor="black",
        s=220,
        zorder=5
    )

    opt_mask = (Z_feasible == f_opt)
    x_opt_all = X[opt_mask]
    y_opt_all = Y[opt_mask]

    ax.set_title(f"{P.kind}:  ≈ f({x_opt:. 2f}, {y_opt:.2f}) = {f_opt:.2f}")
    ax.set_xlabel(d1.name)
    ax.set_ylabel(d2.name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    ax.scatter(
        x_opt_all, y_opt_all,
        color="gold",
        edgecolor="goldenrod",
        s=40,
        zorder=5
    )

    plt.show()
