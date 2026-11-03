"""Coordinate sampling and validation for two-dimensional Graphikon views."""

import numpy as np

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
