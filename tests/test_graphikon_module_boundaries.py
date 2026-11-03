"""Verify Graphikon's module ownership and compatibility after extraction."""
import inspect
import unittest

import numpy as np
import sympy as sym

from Wizard.Graphikon import grid, evaluation, constraints, coloring, plots, utils
from Wizard import Decision, Optimization


class GraphikonModuleBoundaryTests(unittest.TestCase):
    def test_old_utility_symbols_are_import_aliases_not_copies(case):
        canonical = {
            "create_meshgrid": grid.create_meshgrid,
            "_check_grid_structure": grid._check_grid_structure,
            "_get_coordinate_values": grid._get_coordinate_values,
            "_evaluate_expression": evaluation._evaluate_expression,
            "evaluate_objective": evaluation.evaluate_objective,
            "find_optimal_point": evaluation.find_optimal_point,
            "compute_gradient_field": evaluation.compute_gradient_field,
            "compute_constraint_masks": constraints.compute_constraint_masks,
            "get_feasible_mask": constraints.get_feasible_mask,
            "affine_equality_band": constraints.affine_equality_band,
            "compute_constraint_satisfaction_ratio": constraints.compute_constraint_satisfaction_ratio,
            "compute_constraint_background": coloring.compute_constraint_background,
            "_angles_to_hsv": coloring._angles_to_hsv,
            "apply_feasible_coloring": coloring.apply_feasible_coloring,
            "convert_hsv_to_rgb": coloring.convert_hsv_to_rgb,
            "normalize_vector_field": plots.normalize_vector_field,
        }
        for name, implementation in canonical.items():
            with case.subTest(name=name):
                case.assertIs(getattr(utils, name), implementation)
                case.assertNotEqual(implementation.__module__, utils.__name__)

    def test_modules_follow_one_way_dependencies(case):
        # Plotting may import maths and colours. None of the leaf modules
        # may import plotting or the compatibility shim.
        for module in (grid, evaluation, constraints, coloring):
            source = inspect.getsource(module)
            with case.subTest(module=module.__name__):
                case.assertNotIn("from .plots import", source)
                case.assertNotIn("from .utils import", source)
        case.assertNotIn("from .utils import", inspect.getsource(plots))

    def test_direct_modules_evaluate_same_problem(case):
        x = Decision("x", lower=-2, upper=3)
        y = Decision("y", lower=-3, upper=4)
        problem = Optimization([x, y])
        problem[x + y <= 3]
        problem[x >= -1]
        problem.set_objetivo((x-1)**2 + (y-2)**2, kind="min")

        X, Y, _ = grid.create_meshgrid(problem, resolution=13)
        values = evaluation.evaluate_objective(problem, X, Y)
        masks = constraints.compute_constraint_masks(problem, X, Y)
        feasible = constraints.get_feasible_mask(masks, X.shape)
        hue, saturation, brightness = coloring.compute_constraint_background(X, masks)
        rgb = coloring.convert_hsv_to_rgb(
            *coloring.apply_feasible_coloring(hue, saturation, brightness, feasible)
        )
        _, optimum, _ = evaluation.find_optimal_point(values, feasible, "min")
        case.assertEqual(X.shape, (13, 13))
        case.assertEqual(len(masks), 2)
        case.assertEqual(rgb.shape, (13, 13, 3))
        case.assertTrue(np.isfinite(rgb).all())
        case.assertGreaterEqual(optimum, 0)

    def test_unsupported_equality_not_a_feasibility_band(case):
        x, y = sym.symbols("x y")
        X, Y = np.meshgrid([0., 1., 2.], [0., 1., 2.])
        model = type("Problem", (), {
            "restrictions": [sym.Eq(x+y, sym.Rational(6, 5))],
            "symbols": lambda _: [x, y],
        })()
        mask = constraints.compute_constraint_masks(model, X, Y)[0]
        band = constraints.affine_equality_band(model.restrictions[0], [x, y], X, Y)
        case.assertFalse(mask.any())
        case.assertTrue(band.any())


if __name__ == "__main__":
    unittest.main()
