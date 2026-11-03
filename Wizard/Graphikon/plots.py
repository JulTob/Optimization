"""
Plotting functions for optimization visualization. 

Provides three main visualization types:
- plot(): Scatter plot with constraint coloring
- field(): Vector field showing gradient direction
- field_contour(): Contour plot with vector field
"""

import numpy as np
import matplotlib.pyplot as plt

from .grid import create_meshgrid
from .evaluation import (
    evaluate_objective, compute_gradient_field, find_optimal_point
)
from .constraints import (
    compute_constraint_masks, get_feasible_mask,
    compute_constraint_satisfaction_ratio
)
from .coloring import (
    compute_constraint_background, apply_feasible_coloring, convert_hsv_to_rgb
)


def _plot_style(palette):
    """Resolve the existing grayscale/opacity defaults."""
    return ("gray", 0.7) if palette is None else (palette, 1.0)


def _sample_surface(problem, resolution):
    """Prepare one call's grid, objective, feasibility and visual background.

    The returned arrays live only in the requesting plot call; the drawing
    band of an equality is never used as a feasibility certificate.
    """
    X, Y, bounds = create_meshgrid(problem, resolution)
    Z = evaluate_objective(problem, X, Y)
    masks = compute_constraint_masks(problem, X, Y)
    feasible_mask = get_feasible_mask(masks, X.shape)

    h_bg, s_bg, v_bg = compute_constraint_background(X, masks)
    h_bg, s_bg, v_bg = apply_feasible_coloring(
        h_bg, s_bg, v_bg, feasible_mask
    )
    background = convert_hsv_to_rgb(h_bg, s_bg, v_bg)
    return X, Y, Z, bounds, masks, feasible_mask, background


def _begin_figure(background, bounds):
    """Draw the common feasible/violated-region background."""
    _, ax = plt.subplots(figsize=(8, 8))
    xmin, xmax, ymin, ymax = bounds
    ax.imshow(background, extent=[xmin, xmax, ymin, ymax], origin="lower")
    return ax


def _draw_gradient(ax, problem, X, Y, Z, bounds, masks,
                   resolution, palette, opacity):
    """Draw the common gradient-arrow layer for field and field_contour."""
    GX, GY = compute_gradient_field(problem, X, Y)

    ratio = compute_constraint_satisfaction_ratio(masks)
    size_factor = 0.3 + 0.7 * ratio

    xmin, xmax, ymin, ymax = bounds
    base_len = 0.03 * max(xmax - xmin, ymax - ymin)
    U, V = normalize_vector_field(GX, GY, base_len, size_factor)

    step = max(1, resolution // 25)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        Z[::step, ::step],
        cmap=palette,
        alpha=opacity,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )


def _end_figure(ax, problem, X, Y, Z, feasible_mask, bounds):
    """Mark the sampled optimum, label axes and display the figure."""
    _mark_optimal_point(ax, X, Y, Z, feasible_mask, problem)
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlabel(problem.variables[0].name)
    ax.set_ylabel(problem.variables[1].name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot(P, resolution=200, palette=None, numeric="Real"):
    """Scatter objective samples on the feasible/violated-region background.

    The numeric argument is retained for API compatibility; variable
    domains determine the grid's numeric types.
    """
    palette, opacity = _plot_style(palette)
    X, Y, Z, bounds, masks, feasible_mask, background = _sample_surface(
        P, resolution
    )
    ax = _begin_figure(background, bounds)
    ax.scatter(
        X.flatten(), Y.flatten(),
        c=Z.flatten(), cmap=palette,
        edgecolors="none", alpha=opacity,
    )
    _end_figure(ax, P, X, Y, Z, feasible_mask, bounds)


def field(P, resolution=200, palette=None, numeric="Real"):
    """Render the objective's normalized ascent-gradient vector field."""
    palette, opacity = _plot_style(palette)
    X, Y, Z, bounds, masks, feasible_mask, background = _sample_surface(
        P, resolution
    )
    ax = _begin_figure(background, bounds)
    _draw_gradient(
        ax, P, X, Y, Z, bounds, masks, resolution, palette, opacity
    )
    _end_figure(ax, P, X, Y, Z, feasible_mask, bounds)


def field_contour(P, resolution=200, palette=None, numeric="Real"):
    """Render objective contours over the same gradient vector field."""
    palette, opacity = _plot_style(palette)
    X, Y, Z, bounds, masks, feasible_mask, background = _sample_surface(
        P, resolution
    )
    ax = _begin_figure(background, bounds)
    ax.contour(
        X, Y, Z, levels=10,
        colors="black", alpha=0.5, linewidths=0.8,
    )
    _draw_gradient(
        ax, P, X, Y, Z, bounds, masks, resolution, palette, opacity
    )
    _end_figure(ax, P, X, Y, Z, feasible_mask, bounds)

def _mark_optimal_point(ax, X, Y, Z, feasible_mask, P):
    """
    Helper function to find and mark the optimal point on plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Plot axes
    X, Y : np.ndarray
        Coordinate grids
    Z : np.ndarray
        Objective values
    feasible_mask : np.ndarray
        Feasibility mask
    P : Optimization
        Optimization problem
    """
    opt_idx, opt_val, opt_mask = find_optimal_point(Z, feasible_mask, P.kind)
    
    x_opt = X. flatten()[opt_idx]
    y_opt = Y.flatten()[opt_idx]
    
    # Mark single optimal point
    ax.scatter(
        [x_opt], [y_opt],
        color="gold",
        edgecolor="black",
        s=220,
        zorder=5,
        label=f"{P.kind}:  {opt_val:.2f}"
    )
    
    # Mark all optimal points (in case of ties)
    x_opt_all = X[opt_mask]
    y_opt_all = Y[opt_mask]
    
    ax.scatter(
        x_opt_all, y_opt_all,
        color="gold",
        edgecolor="goldenrod",
        s=40,
        zorder=5
    )
    
    ax.set_title(
        f"{P.kind.capitalize()}: f({x_opt:.2f}, {y_opt:.2f}) = {opt_val:.2f}"
    )


def normalize_vector_field(GX, GY, base_len, size_factor):
    """Normalize arrows while giving zero-gradient points zero arrows."""
    norm = np.hypot(GX, GY)
    safe_norm = np.where(norm == 0, 1.0, norm)
    return (
        GX / safe_norm * base_len * size_factor,
        GY / safe_norm * base_len * size_factor,
    )
