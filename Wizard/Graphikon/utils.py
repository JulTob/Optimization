"""Reusable numerical helpers for two-dimensional optimization plots.

Two different questions are kept separate:
- Feasibility masks test the constraints at sampled points.
- Equality bands approximate the *drawing* of an affine equality on a grid.

A visually nearby point is not thereby certified as a feasible optimizer.
"""

import numpy as np
import sympy as sym
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def _check_grid_structure(X, Y):
    """Validate two nonempty 2D coordinate arrays with identical shapes."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2:
        raise ValueError(f"X must have two dimensions; received {X.ndim}.")
    if Y.ndim != 2:
        raise ValueError(f"Y must have two dimensions; received {Y.ndim}.")
    if X.shape != Y.shape:
        raise ValueError(f"Grid shapes differ: X{X.shape} versus Y{Y.shape}.")
    if X.size == 0:
        raise ValueError("Coordinate grids must not be empty.")
    return X, Y


def _evaluate_expression(expression, symbols, X, Y):
    """Evaluate a SymPy expression on a grid, including constants."""
    evaluator = sym.lambdify(symbols, expression, modules="numpy")
    return np.broadcast_to(np.asarray(evaluator(X, Y), dtype=float), X.shape)


def create_meshgrid(P, resolution=200):
    """Return X, Y, (xmin, xmax, ymin, ymax) for two bounded decisions.

    ``resolution`` counts samples per axis for real-valued variables;
    integer-valued variables instead include every integer in their domain.
    """
    if len(P.variables) != 2:
        raise ValueError("Graphikon's grid requires exactly two decisions.")
    if (not isinstance(resolution, (int, np.integer))
            or isinstance(resolution, (bool, np.bool_))
            or resolution < 2):
        raise ValueError("resolution must be an integer of at least 2.")

    x_decision, y_decision = P.variables
    xmin, xmax = x_decision.lower, x_decision.upper
    ymin, ymax = y_decision.lower, y_decision.upper

    x_values = _get_coordinate_values(x_decision, xmin, xmax, resolution)
    y_values = _get_coordinate_values(y_decision, ymin, ymax, resolution)
    X, Y = np.meshgrid(x_values, y_values)
    return X, Y, (xmin, xmax, ymin, ymax)


def _get_coordinate_values(decision, lower, upper, resolution):
    """Sample the finite declared domain of one decision variable."""
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f"{decision.name} needs finite, increasing bounds.")

    if decision.numeric == "Real":
        return np.linspace(lower, upper, resolution)
    if decision.numeric == "Integer":
        if int(lower) != lower or int(upper) != upper:
            raise ValueError(f"{decision.name} has non-integer integer bounds.")
        return np.arange(int(lower), int(upper) + 1)
    raise ValueError(f"Unsupported numeric type: {decision.numeric}")


def evaluate_objective(P, X, Y):
    """Return objective values with exactly the shape of the coordinate grid."""
    X, Y = _check_grid_structure(X, Y)
    if P.objective is None:
        raise ValueError("No objective function defined.")
    return _evaluate_expression(P.objective, P.symbols(), X, Y)


def compute_constraint_masks(P, X, Y, atol=0.0):
    """Return pointwise constraint-satisfaction masks.

    A True value means that the *sampled point* satisfies a constraint.
    ``atol`` permits only explicit numerical slack for equality residuals;
    it is NOT a grid-resolution-dependent drawing tolerance.
    """
    X, Y = _check_grid_structure(X, Y)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")

    masks = []
    for relation in P.restrictions:
        supported = (
            sym.LessThan, sym.GreaterThan, sym.Equality,
            sym.StrictLessThan, sym.StrictGreaterThan,
        )
        if not isinstance(relation, supported):
            raise ValueError(f"Unsupported constraint: {relation}")
        residual = _evaluate_expression(
            relation.lhs - relation.rhs, P.symbols(), X, Y
        )
        if isinstance(relation, sym.StrictLessThan):
            mask = residual < 0
        elif isinstance(relation, sym.LessThan):
            mask = residual <= 0
        elif isinstance(relation, sym.StrictGreaterThan):
            mask = residual > 0
        elif isinstance(relation, sym.GreaterThan):
            mask = residual >= 0
        elif isinstance(relation, sym.Equality):
            mask = np.isclose(residual, 0.0, rtol=0.0, atol=atol)
        else:
            raise ValueError(f"Unsupported constraint: {relation}")
        masks.append(mask)
    return masks


def affine_equality_band(relation, symbols, X, Y):
    """Approximate an affine equality for DISPLAY, not for feasibility.

    For g(x,y)=a*x+b*y+c, use |g| <= (|a|*hx+|b|*hy)/2,
    where hx, hy are the maximum adjacent grid steps.
    This conservative sampling band is not a numerical error bound.
    """
    X, Y = _check_grid_structure(X, Y)
    if not isinstance(relation, sym.Equality):
        raise ValueError("Expected a SymPy equality (Eq).")
    if len(symbols) != 2 or min(X.shape) < 2:
        raise ValueError("An affine equality band needs a 2D grid of at least 2x2.")

    x, y = symbols
    residual = relation.lhs - relation.rhs
    coefficient_x = sym.diff(residual, x)
    coefficient_y = sym.diff(residual, y)
    if (coefficient_x.has(x, y) or coefficient_y.has(x, y)
            or (residual - coefficient_x * x - coefficient_y * y).has(x, y)):
        raise ValueError("The equality must be affine in the plotted variables.")

    hx = np.max(np.abs(np.diff(X, axis=1)))
    hy = np.max(np.abs(np.diff(Y, axis=0)))
    grid_tolerance = (
        abs(float(coefficient_x)) * hx + abs(float(coefficient_y)) * hy
    ) / 2.0
    values = _evaluate_expression(residual, symbols, X, Y)
    return np.abs(values) <= grid_tolerance


def get_feasible_mask(masks, shape):
    """Intersection of pointwise constraint masks; all True if no constraints."""
    feasible_mask = np.ones(shape, dtype=bool)
    for mask in masks:
        feasible_mask &= mask
    return feasible_mask


def compute_constraint_background(X, masks):
    """Build HSV colors for violated, not satisfied, constraints."""
    colors_rgb = np.array([
        [0.0, 1.0, 0.0],  # Green
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ], dtype=float)
    colors_hsv = rgb_to_hsv(colors_rgb)
    shape = np.shape(X)
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)

    for index, satisfied in enumerate(masks):
        violated = ~satisfied
        hue = colors_hsv[index % len(colors_hsv), 0]
        angle = 2.0 * np.pi * hue
        sum_x[violated] += np.cos(angle)
        sum_y[violated] += np.sin(angle)
        count[violated] += 1.0
    return _angles_to_hsv(sum_x, sum_y, count, shape)


def _angles_to_hsv(sum_x, sum_y, count, shape):
    """Convert aggregated color directions into HSV channel arrays."""
    hue = np.zeros(shape, dtype=float)
    saturation = np.zeros(shape, dtype=float)
    brightness = np.zeros(shape, dtype=float)

    active = count > 0
    angles = np.arctan2(sum_y[active], sum_x[active])
    hue[active] = (angles % (2.0 * np.pi)) / (2.0 * np.pi)
    saturation[active] = 1.0
    brightness[active] = np.clip(
        0.4 + 0.15 * (count[active] - 1.0), 0.3, 0.8
    )
    return hue, saturation, brightness


def apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask):
    """Display feasible samples in white without modifying input arrays."""
    hue, saturation, brightness = h_bg.copy(), s_bg.copy(), v_bg.copy()
    saturation[feasible_mask] = 0.0
    brightness[feasible_mask] = 1.0
    return hue, saturation, brightness


def convert_hsv_to_rgb(h_bg, s_bg, v_bg):
    """Combine hue, saturation and brightness into one RGB image."""
    return hsv_to_rgb(np.stack([h_bg, s_bg, v_bg], axis=-1))


def find_optimal_point(Z, feasible_mask, kind="min"):
    """Find the best finite, sampled feasible point; not a continuous proof."""
    Z = np.asarray(Z, dtype=float)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    if Z.shape != feasible_mask.shape:
        raise ValueError("Objective and feasibility masks must have equal shape.")
    if kind not in ("min", "max"):
        raise ValueError(f"Unsupported optimization direction: {kind}")

    eligible = feasible_mask & np.isfinite(Z)
    if not eligible.any():
        raise ValueError("No feasible finite samples on this grid.")
    Z_feasible = np.where(eligible, Z, np.nan)
    optimum_index = (
        np.nanargmin(Z_feasible) if kind == "min"
        else np.nanargmax(Z_feasible)
    )
    optimum_value = Z_feasible.ravel()[optimum_index]
    optimum_mask = Z_feasible == optimum_value
    return int(optimum_index), float(optimum_value), optimum_mask


def compute_gradient_field(P, X, Y):
    """Compute both gradient components, also for constant derivatives."""
    X, Y = _check_grid_structure(X, Y)
    if P.objective is None:
        raise ValueError("No objective function defined.")
    derivative_x, derivative_y = (
        sym.diff(P.objective, symbol) for symbol in P.symbols()
    )
    return (
        _evaluate_expression(derivative_x, P.symbols(), X, Y),
        _evaluate_expression(derivative_y, P.symbols(), X, Y),
    )


def normalize_vector_field(GX, GY, base_len, size_factor):
    """Normalize arrows while giving zero-gradient points zero arrows."""
    norm = np.hypot(GX, GY)
    safe_norm = np.where(norm == 0, 1.0, norm)
    return (
        GX / safe_norm * base_len * size_factor,
        GY / safe_norm * base_len * size_factor,
    )


def compute_constraint_satisfaction_ratio(masks, shape=None):
    """Fraction of satisfied constraints; 1.0 when there are none.

    With no constraints, a scalar 1.0 broadcasts to the current Graphikon
    field/field_contour callers. Pass shape to obtain an array instead.
    """
    if not masks:
        return 1.0 if shape is None else np.ones(shape, dtype=float)
    return np.mean(np.stack(masks, axis=0), axis=0)