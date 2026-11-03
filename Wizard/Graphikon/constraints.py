"""Constraint evaluation and feasible-region operations.

An affine equality band is for display only, not a feasibility certificate.
"""

import numpy as np
import sympy as sym

from .grid import _check_grid_structure
from .evaluation import _evaluate_expression

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


def compute_constraint_satisfaction_ratio(masks, shape=None):
    """Fraction of satisfied constraints; 1.0 when there are none.

    With no constraints, a scalar 1.0 broadcasts to the current Graphikon
    field/field_contour callers. Pass shape to obtain an array instead.
    """
    if not masks:
        return 1.0 if shape is None else np.ones(shape, dtype=float)
    return np.mean(np.stack(masks, axis=0), axis=0)
