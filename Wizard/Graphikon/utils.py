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
    """
    Require two equally sized (shape two-dimensional coordinate arrays.

    A shape means the same number of dimensions, and the same number of elements per dimension.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.ndim == 2, f"X must be 2D, got {X.ndim}D."
    assert Y.ndim == 2, f"Y must be 2D, got {Y.ndim}D."

    assert X.shape == Y.shape, (
        f"Shape mismatch: X{X.shape} != Y{Y.shape}."
    )

    assert X.size > 0, "Coordinate arrays cannot be empty."

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
    X, Y = _check_grid(X, Y)
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
