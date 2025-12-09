"""
Utility functions for visualization of optimization problems. 

This module contains helper functions that abstract away common visualization
tasks, following the DRY (Don't Repeat Yourself) principle.
"""

import numpy as np
import sympy as sym
from matplotlib. colors import rgb_to_hsv, hsv_to_rgb


def create_meshgrid(P, resolution=200):
    """
    Create coordinate grids for 2D visualization.
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    resolution : int, optional
        Number of points per axis (default 200)
        
    Returns
    -------
    tuple
        (X, Y, bounds) where X and Y are meshgrids and bounds = (xmin, xmax, ymin, ymax)
    """
    xs = P.symbols()
    d1, d2 = P.variables
    xmin, xmax = 0, d1.upper
    ymin, ymax = 0, d2.upper
    
    x_vals = _get_coordinate_values(d1, xmin, xmax, resolution)
    y_vals = _get_coordinate_values(d2, ymin, ymax, resolution)
    
    X, Y = np.meshgrid(x_vals, y_vals)
    return X, Y, (xmin, xmax, ymin, ymax)


def _get_coordinate_values(var, vmin, vmax, resolution):
    """
    Get coordinate values based on variable type (Real or Integer).
    
    Parameters
    ----------
    var : Decision
        Decision variable
    vmin :  float
        Minimum value
    vmax : float
        Maximum value
    resolution : int
        Number of points
        
    Returns
    -------
    np.ndarray
        Coordinate values
    """
    if var.numeric == "Real":
        return np.linspace(vmin, vmax, resolution)
    elif var.numeric == "Integer": 
        return np.arange(int(vmin), int(vmax) + 1)
    else:
        raise ValueError(f"Numeric type not supported: {var.numeric}")


def evaluate_objective(P, X, Y):
    """
    Evaluate objective function on grid points.
    
    Parameters
    ----------
    P : Optimization
        Optimization problem
    X : np.ndarray
        X-coordinate grid
    Y : np.ndarray
        Y-coordinate grid
        
    Returns
    -------
    np.ndarray
        Objective function values at grid points
    """
    xs = P.symbols()
    if P.objective is None:
        raise ValueError("No objective function defined.")
    f_num = sym.lambdify(xs, P.objective, "numpy")
    return f_num(X, Y)


def compute_constraint_masks(P, X, Y):
    """
    Compute boolean masks for each constraint.
    
    A mask is True where the constraint is satisfied (g(x,y) <= 0).
    
    Parameters
    ----------
    P : Optimization
        Optimization problem
    X : np.ndarray
        X-coordinate grid
    Y : np.ndarray
        Y-coordinate grid
        
    Returns
    -------
    list[np.ndarray]
        List of boolean masks, one per constraint
    """
    xs = P.symbols()
    masks = []
    for r in P.restrictions:
        g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
        G = g_num(X, Y)
        masks.append(G <= 0)
    return masks


def get_feasible_mask(masks, shape):
    """
    Combine constraint masks into single feasibility mask.
    
    A point is feasible if it satisfies ALL constraints.
    
    Parameters
    ----------
    masks : list[np.ndarray]
        List of constraint masks
    shape : tuple
        Shape of grid
        
    Returns
    -------
    np.ndarray
        Boolean mask where feasible_mask[i,j] = True iff point is feasible
    """
    feasible_mask = np.ones(shape, dtype=bool)
    for mask in masks:
        feasible_mask &= mask
    return feasible_mask


def compute_constraint_background(X, masks):
    """
    Compute HSV background colors for constraint violation regions.
    
    Uses color cycling to show which constraints are violated.
    When multiple constraints are violated, colors blend based on 
    the constraint violation directions.
    
    Parameters
    ----------
    X : np.ndarray
        X-coordinate grid (used for shape only)
    masks : list[np.ndarray]
        List of constraint masks
        
    Returns
    -------
    tuple
        (h_bg, s_bg, v_bg) - HSV components
    """
    colors_rgb = np.array([
        [0.0, 1.0, 0.0],  # Green
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ])
    colors_hsv = rgb_to_hsv(colors_rgb)
    
    shape = X.shape
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)
    
    # For each constraint, accumulate unit vectors in HSV color direction
    for i, mask in enumerate(masks):
        h, s, v = colors_hsv[i % len(colors_hsv)]
        angle = 2 * np.pi * h
        sum_x[mask] += np.cos(angle)
        sum_y[mask] += np.sin(angle)
        count[mask] += 1.0
    
    return _angles_to_hsv(sum_x, sum_y, count, shape)


def _angles_to_hsv(sum_x, sum_y, count, shape):
    """
    Convert constraint angle sums to HSV color space. 
    
    Parameters
    ----------
    sum_x, sum_y : np.ndarray
        Accumulated unit vector components
    count : np.ndarray
        Number of violated constraints at each point
    shape : tuple
        Grid shape
        
    Returns
    -------
    tuple
        (h_bg, s_bg, v_bg) - HSV components
    """
    h_bg = np.zeros(shape, dtype=float)
    s_bg = np.zeros(shape, dtype=float)
    v_bg = np.zeros(shape, dtype=float)
    
    active = count > 0
    
    # Compute hue from angle of accumulated vectors
    angles = np.arctan2(sum_y[active], sum_x[active])
    h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)
    
    # Saturation indicates constraint violation
    s_bg[active] = 1.0
    
    # Value based on number of violated constraints
    v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)
    
    return h_bg, s_bg, v_bg


def apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask):
    """
    Apply white coloring to feasible regions.
    
    Feasible regions are shown in white (s=0, v=1).
    
    Parameters
    ----------
    h_bg, s_bg, v_bg :  np.ndarray
        HSV components
    feasible_mask : np. ndarray
        Boolean mask of feasible regions
        
    Returns
    -------
    tuple
        (h_bg, s_bg, v_bg) - Updated HSV components
    """
    h_bg_copy = h_bg.copy()
    s_bg_copy = s_bg.copy()
    v_bg_copy = v_bg.copy()
    
    s_bg_copy[feasible_mask] = 0.0
    v_bg_copy[feasible_mask] = 1.0
    
    return h_bg_copy, s_bg_copy, v_bg_copy


def convert_hsv_to_rgb(h_bg, s_bg, v_bg):
    """
    Convert HSV arrays to RGB for rendering.
    
    Parameters
    ----------
    h_bg, s_bg, v_bg : np.ndarray
        HSV components
        
    Returns
    -------
    np.ndarray
        RGB image array
    """
    hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
    return hsv_to_rgb(hsv_bg)


def find_optimal_point(Z, feasible_mask, kind="min"):
    """
    Find optimal point in feasible region. 
    
    Parameters
    ----------
    Z : np.ndarray
        Objective function values
    feasible_mask : np. ndarray
        Boolean mask of feasible region
    kind : str, optional
        "min" for minimization, "max" for maximization
        
    Returns
    -------
    tuple
        (opt_idx, opt_val, opt_mask) where:
        - opt_idx:  linear index of optimal point
        - opt_val: objective value at optimal point
        - opt_mask: boolean mask of all optimal points (in case of ties)
    """
    Z_feasible = np.where(feasible_mask, Z, np.nan)
    
    if kind == "max":
        opt_idx = np.nanargmax(Z_feasible)
    else:
        opt_idx = np.nanargmin(Z_feasible)
    
    opt_val = Z_feasible. flatten()[opt_idx]
    opt_mask = (Z_feasible == opt_val)
    
    return opt_idx, opt_val, opt_mask


def compute_gradient_field(P, X, Y):
    """
    Compute gradient vector field of objective function.
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    X : np.ndarray
        X-coordinate grid
    Y :  np.ndarray
        Y-coordinate grid
        
    Returns
    -------
    tuple
        (GX, GY) - Gradient components
    """
    xs = P.symbols()
    grad = [sym.diff(P.objective, var) for var in xs]
    
    gx_num = sym.lambdify(xs, grad[0], "numpy")
    gy_num = sym.lambdify(xs, grad[1], "numpy")
    
    GX = gx_num(X, Y)
    GY = gy_num(X, Y)
    
    # Handle scalar outputs (constant gradient)
    if np.isscalar(GX):
        GX = np.full_like(X, GX)
    if np.isscalar(GY):
        GY = np.full_like(Y, GY)
    
    return GX, GY


def normalize_vector_field(GX, GY, base_len, size_factor):
    """
    Normalize and scale gradient vector field for visualization.
    
    Parameters
    ----------
    GX, GY : np.ndarray
        Gradient components
    base_len : float
        Base length scale for arrows
    size_factor : np.ndarray
        Per-point scaling factors (based on constraint satisfaction)
        
    Returns
    -------
    tuple
        (U, V) - Scaled gradient components ready for quiver plot
    """
    norm = np.hypot(GX, GY)
    norm[norm == 0] = 1. 0
    
    U = GX / norm * base_len * size_factor
    V = GY / norm * base_len * size_factor
    
    return U, V


def compute_constraint_satisfaction_ratio(masks):
    """
    Compute ratio of satisfied constraints at each point.
    
    Used to scale arrow size based on feasibility.
    
    Parameters
    ----------
    masks : list[np.ndarray]
        List of constraint masks
        
    Returns
    -------
    np. ndarray
        Ratio array (0 to 1) indicating constraint satisfaction
    """
    n_constraints = len(masks)
    
    if n_constraints == 0:
        # No constraints - all points equally feasible
        return np.ones_like(masks[0], dtype=float) if masks else None
    
    satisfied_count = np.zeros_like(masks[0], dtype=float)
    for mask in masks:
        satisfied_count += mask
    
    return satisfied_count / n_constraints
