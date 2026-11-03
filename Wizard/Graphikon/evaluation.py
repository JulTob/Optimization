"""Evaluate SymPy objectives and derivatives on sampled coordinates."""

import numpy as np
import sympy as sym

from .grid import _check_grid_structure

def _evaluate_expression(expression, symbols, X, Y):
    """Evaluate a SymPy expression on a grid, including constants."""
    evaluator = sym.lambdify(symbols, expression, modules="numpy")
    return np.broadcast_to(np.asarray(evaluator(X, Y), dtype=float), X.shape)


def evaluate_objective(P, X, Y):
    """Return objective values with exactly the shape of the coordinate grid."""
    X, Y = _check_grid_structure(X, Y)
    if P.objective is None:
        raise ValueError("No objective function defined.")
    return _evaluate_expression(P.objective, P.symbols(), X, Y)


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
