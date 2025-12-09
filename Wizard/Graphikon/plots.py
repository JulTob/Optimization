"""
Plotting functions for optimization visualization. 

Provides three main visualization types:
- plot(): Scatter plot with constraint coloring
- field(): Vector field showing gradient direction
- field_contour(): Contour plot with vector field
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from .utils import (
    create_meshgrid,
    evaluate_objective,
    compute_constraint_masks,
    get_feasible_mask,
    compute_constraint_background,
    apply_feasible_coloring,
    convert_hsv_to_rgb,
    find_optimal_point,
    compute_gradient_field,
    normalize_vector_field,
    compute_constraint_satisfaction_ratio,
)


def plot(P, resolution=200, palette=None, numeric="Real"):
    """
    Create a scatter plot visualization of the optimization problem. 
    
    Shows: 
    - Constraint violation regions (colored by violated constraint)
    - Feasible region (white)
    - Objective function values (point colors)
    - Optimal point (gold star)
    
    Parameters
    ----------
    P : Optimization
        Optimization problem
    resolution :  int, optional
        Grid resolution (default 200)
    palette : str, optional
        Matplotlib colormap for objective (default 'gray')
    numeric : str, optional
        Numeric type (default 'Real')
    """
    if palette is None:
        palette = "gray"
        dot_alpha = 0.7
    else:
        dot_alpha = 1.0
    
    # Prepare data
    X, Y, bounds = create_meshgrid(P, resolution)
    Z = evaluate_objective(P, X, Y)
    masks = compute_constraint_masks(P, X, Y)
    feasible_mask = get_feasible_mask(masks, X. shape)
    
    # Compute colors
    h_bg, s_bg, v_bg = compute_constraint_background(X, masks)
    h_bg, s_bg, v_bg = apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask)
    rgb_bg = convert_hsv_to_rgb(h_bg, s_bg, v_bg)
    
    # Create figure
    fig, ax = plt. subplots(figsize=(8, 8))
    
    xmin, xmax, ymin, ymax = bounds
    
    # Draw background
    ax.imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )
    
    # Draw objective function as scatter
    ax.scatter(
        X.flatten(), Y.flatten(),
        c=Z.flatten(),
        cmap=palette,
        edgecolors="none",
        alpha=dot_alpha
    )
    
    # Find and mark optimal point
    _mark_optimal_point(ax, X, Y, Z, feasible_mask, P)
    
    # Labels and formatting
    ax.set_xlabel(P.variables[0].name)
    ax.set_ylabel(P.variables[1].name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    
    plt.show()


def field(P, resolution=200, palette=None, numeric="Real"):
    """
    Create a vector field visualization of the optimization problem.
    
    Shows:
    - Constraint violation regions
    - Feasible region
    - Gradient vector field (indicating direction of steepest ascent)
    - Optimal point
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    resolution : int, optional
        Grid resolution (default 200)
    palette : str, optional
        Matplotlib colormap for objective (default 'gray')
    numeric : str, optional
        Numeric type (default 'Real')
    """
    if palette is None: 
        palette = "gray"
        dot_alpha = 0.7
    else:
        dot_alpha = 1.0
    
    # Prepare data
    X, Y, bounds = create_meshgrid(P, resolution)
    Z = evaluate_objective(P, X, Y)
    masks = compute_constraint_masks(P, X, Y)
    feasible_mask = get_feasible_mask(masks, X. shape)
    
    # Compute colors
    h_bg, s_bg, v_bg = compute_constraint_background(X, masks)
    h_bg, s_bg, v_bg = apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask)
    rgb_bg = convert_hsv_to_rgb(h_bg, s_bg, v_bg)
    
    # Compute gradient field
    GX, GY = compute_gradient_field(P, X, Y)
    
    # Compute constraint satisfaction for sizing arrows
    ratio = compute_constraint_satisfaction_ratio(masks)
    size_factor = 0.3 + 0.7 * ratio
    
    # Normalize gradient field
    xmin, xmax, ymin, ymax = bounds
    base_len = 0.03 * max(xmax - xmin, ymax - ymin)
    U, V = normalize_vector_field(GX, GY, base_len, size_factor)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw background
    ax.imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )
    
    # Draw vector field (subsample for clarity)
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
    
    # Find and mark optimal point
    _mark_optimal_point(ax, X, Y, Z, feasible_mask, P)
    
    # Labels and formatting
    ax. set_xlabel(P.variables[0].name)
    ax.set_ylabel(P.variables[1].name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    
    plt.show()


def field_contour(P, resolution=200, palette=None, numeric="Real"):
    """
    Create a contour plot with vector field visualization.
    
    Shows:
    - Constraint violation regions
    - Feasible region
    - Contour lines of objective function
    - Gradient vector field
    - Optimal point
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    resolution : int, optional
        Grid resolution (default 200)
    palette : str, optional
        Matplotlib colormap for objective (default 'gray')
    numeric : str, optional
        Numeric type (default 'Real')
    """
    if palette is None: 
        palette = "gray"
        dot_alpha = 0.7
    else:
        dot_alpha = 1.0
    
    # Prepare data
    X, Y, bounds = create_meshgrid(P, resolution)
    Z = evaluate_objective(P, X, Y)
    masks = compute_constraint_masks(P, X, Y)
    feasible_mask = get_feasible_mask(masks, X. shape)
    
    # Compute colors
    h_bg, s_bg, v_bg = compute_constraint_background(X, masks)
    h_bg, s_bg, v_bg = apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask)
    rgb_bg = convert_hsv_to_rgb(h_bg, s_bg, v_bg)
    
    # Compute gradient field
    GX, GY = compute_gradient_field(P, X, Y)
    
    # Compute constraint satisfaction for sizing arrows
    ratio = compute_constraint_satisfaction_ratio(masks)
    size_factor = 0.3 + 0.7 * ratio
    
    # Normalize gradient field
    xmin, xmax, ymin, ymax = bounds
    base_len = 0.03 * max(xmax - xmin, ymax - ymin)
    U, V = normalize_vector_field(GX, GY, base_len, size_factor)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw background
    ax. imshow(
        rgb_bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )
    
    # Draw contour lines
    ax.contour(
        X, Y, Z,
        levels=10,
        colors='black',
        alpha=0.5,
        linewidths=0.8
    )
    
    # Draw vector field (subsample for clarity)
    step = max(1, resolution // 25)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[:: step, ::step],
        Z[::step, ::step],
        cmap=palette,
        alpha=dot_alpha,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004
    )
    
    # Find and mark optimal point
    _mark_optimal_point(ax, X, Y, Z, feasible_mask, P)
    
    # Labels and formatting
    ax.set_xlabel(P.variables[0].name)
    ax.set_ylabel(P.variables[1].name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    
    plt.show()


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
