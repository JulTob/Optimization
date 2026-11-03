"""Compatibility imports for Graphikon's former generic utilities.

The implementations now live in purpose-specific modules. These aliases are
temporary to protect existing notebook imports; new code should import from
grid, evaluation, constraints, coloring or plots directly.
"""

from .grid import _check_grid_structure, create_meshgrid, _get_coordinate_values
from .evaluation import (
    _evaluate_expression, evaluate_objective,
    find_optimal_point, compute_gradient_field,
)
from .constraints import (
    compute_constraint_masks, affine_equality_band,
    get_feasible_mask, compute_constraint_satisfaction_ratio,
)
from .coloring import (
    compute_constraint_background, _angles_to_hsv,
    apply_feasible_coloring, convert_hsv_to_rgb,
)
from .plots import normalize_vector_field
