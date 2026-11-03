"""Graphikon numerical helpers: contracts, edge cases and mathematical meaning."""
import unittest
from types import SimpleNamespace

import numpy as np
import sympy as sym

from Wizard.Graphikon import utils as u


def decision(name, lower=0, upper=5, numeric="Real"):
    return SimpleNamespace(name=name, lower=lower, upper=upper, numeric=numeric)


class GraphikonUtilsTests(unittest.TestCase):
    def setUp(case):
        case.x, case.y = sym.symbols("x y")
        case.problem = SimpleNamespace(
            variables=[decision("x", -5, 5), decision("y", -3, 3)],
            symbols=lambda: [case.x, case.y],
            restrictions=[],
            objective=(case.x - 2)**2 + (case.y - 1)**2,
            kind="min",
        )

    def test_grid_structure_explicit_errors(case):
        for first, second, message in (
            (np.zeros(2), np.zeros((2, 2)), "X must have two dimensions"),
            (np.zeros((2, 2)), np.zeros(2), "Y must have two dimensions"),
            (np.zeros((2, 2)), np.zeros((2, 3)), "Grid shapes differ"),
            (np.zeros((0, 2)), np.zeros((0, 2)), "must not be empty"),
        ):
            with case.subTest(message=message):
                with case.assertRaisesRegex(ValueError, message):
                    u._check_grid_structure(first, second)

    def test_grid_respects_bounds_and_integer_domain(case):
        X, Y, bounds = u.create_meshgrid(case.problem, resolution=7)
        case.assertEqual(bounds, (-5, 5, -3, 3))
        case.assertEqual(X.shape, (7, 7))
        case.assertEqual((X.min(), Y.min()), (-5, -3))
        case.problem.variables[0].numeric = "Integer"
        X, _, _ = u.create_meshgrid(case.problem, resolution=7)
        case.assertEqual(X.shape[1], 11)
        case.assertEqual(X[0].tolist(), list(range(-5, 6)))

    def test_invalid_bounds_and_resolution(case):
        case.problem.variables[1].upper = np.inf
        with case.assertRaisesRegex(ValueError, "finite"):
            u.create_meshgrid(case.problem)
        case.problem.variables[1].upper = 3
        with case.assertRaisesRegex(ValueError, "resolution"):
            u.create_meshgrid(case.problem, resolution=1)

    def test_constant_objective_and_zero_gradient(case):
        X, Y, _ = u.create_meshgrid(case.problem, 5)
        case.problem.objective = sym.Integer(7)
        Z = u.evaluate_objective(case.problem, X, Y)
        GX, GY = u.compute_gradient_field(case.problem, X, Y)
        case.assertEqual(Z.shape, X.shape)
        case.assertTrue((Z == 7).all())
        case.assertTrue((GX == 0).all() and (GY == 0).all())
        U, V = u.normalize_vector_field(GX, GY, 1.0, 1.0)
        case.assertTrue((U == 0).all() and (V == 0).all())

    def test_five_relational_types(case):
        X, Y = np.meshgrid([1.0, 2.0, 3.0], [0.0, 1.0])
        case.problem.restrictions = [
            sym.Le(case.x, 2), sym.Lt(case.x, 2),
            sym.Ge(case.x, 2), sym.Gt(case.x, 2),
            sym.Eq(case.x, 2),
        ]
        masks = u.compute_constraint_masks(case.problem, X, Y)
        case.assertEqual([mask[0].tolist() for mask in masks], [
            [True, True, False], [True, False, False],
            [False, True, True], [False, False, True],
            [False, True, False],
        ])
        with case.assertRaisesRegex(ValueError, "atol"):
            u.compute_constraint_masks(case.problem, X, Y, atol=-1)

    def test_equality_band_is_not_feasibility(case):
        X, Y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        equation = sym.Eq(case.x + case.y, sym.Rational(6, 5))
        case.problem.restrictions = [equation]
        exact = u.compute_constraint_masks(case.problem, X, Y)[0]
        band = u.affine_equality_band(equation, case.problem.symbols(), X, Y)
        case.assertFalse(exact.any())
        case.assertTrue(band.any())
        scaled = sym.Eq(1000*case.x + 1000*case.y, 1200)
        np.testing.assert_array_equal(
            band, u.affine_equality_band(scaled, case.problem.symbols(), X, Y)
        )
        with case.assertRaisesRegex(ValueError, "affine"):
            u.affine_equality_band(
                sym.Eq(case.x*case.y, 1), case.problem.symbols(), X, Y
            )

    def test_background_marks_violated_constraints(case):
        X, Y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0])
        case.problem.restrictions = [sym.Ge(case.x, 1)]
        mask = u.compute_constraint_masks(case.problem, X, Y)[0]
        feasible = u.get_feasible_mask([mask], X.shape)
        h, s, v = u.compute_constraint_background(X, [mask])
        h, s, v = u.apply_feasible_coloring(h, s, v, feasible)
        rgb = u.convert_hsv_to_rgb(h, s, v)
        case.assertFalse(np.allclose(rgb[0, 0], 1))
        case.assertTrue(np.allclose(rgb[0, 1], 1))
        case.assertTrue(np.allclose(rgb[0, 2], 1))
        case.assertTrue(u.get_feasible_mask([], X.shape).all())

    def test_sampled_optimum_and_empty_mask(case):
        Z = np.array([[3., 2., 1.], [4., np.nan, 9.]])
        feasible = np.array([[False, True, True], [True, True, True]])
        index, value, winners = u.find_optimal_point(Z, feasible)
        case.assertEqual((index, value), (2, 1.))
        case.assertEqual(winners.sum(), 1)
        with case.assertRaisesRegex(ValueError, "No feasible finite"):
            u.find_optimal_point(Z, np.zeros_like(feasible))
        with case.assertRaisesRegex(ValueError, "direction"):
            u.find_optimal_point(Z, feasible, kind="median")

    def test_constraint_ratio_no_constraints(case):
        case.assertEqual(u.compute_constraint_satisfaction_ratio([]), 1.0)
        np.testing.assert_array_equal(
            u.compute_constraint_satisfaction_ratio([], shape=(2, 2)),
            np.ones((2, 2)),
        )
        masks = [np.array([[True, False]]), np.array([[True, True]])]
        np.testing.assert_allclose(
            u.compute_constraint_satisfaction_ratio(masks), [[1., 0.5]]
        )


if __name__ == "__main__":
    unittest.main()
