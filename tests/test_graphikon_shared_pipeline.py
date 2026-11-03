"""Regression coverage for Graphikon's shared, call-local rendering pipeline."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from Wizard import Decision, Optimization
from Wizard.Graphikon import plots


def example_problem():
    x = Decision("x", lower=-2, upper=4)
    y = Decision("y", lower=-1, upper=5)
    problem = Optimization([x, y])
    problem[x + y <= 6]
    problem.set_objetivo((x - 1)**2 + (y - 3)**2, kind="min")
    return problem


class SharedPlotPipelineTests(unittest.TestCase):
    def tearDown(case):
        plt.close("all")

    def test_surface_computed_once_per_render(case):
        problem = example_problem()
        for name in ("plot", "field", "field_contour"):
            with case.subTest(renderer=name):
                with (
                    patch.object(plots, "create_meshgrid", wraps=plots.create_meshgrid) as mesh,
                    patch.object(plots, "evaluate_objective", wraps=plots.evaluate_objective) as evaluate,
                    patch.object(plots, "compute_constraint_masks", wraps=plots.compute_constraint_masks) as masks,
                    patch.object(plt, "show"),
                ):
                    getattr(problem, name)(resolution=16, palette="magma")
                case.assertEqual(mesh.call_count, 1)
                case.assertEqual(evaluate.call_count, 1)
                case.assertEqual(masks.call_count, 1)
                plt.close("all")

    def test_graph_layers_remain_specific_to_each_renderer(case):
        problem = example_problem()
        for name, expected_quiver, expected_contour in (
            ("plot", 0, 0),
            ("field", 1, 0),
            ("field_contour", 1, 1),
        ):
            with case.subTest(renderer=name):
                with patch.object(plt, "show"):
                    getattr(problem, name)(resolution=16, palette="inferno")
                ax = plt.gcf().axes[0]
                case.assertEqual(len(ax.images), 1)
                case.assertEqual(len(ax.collections) > 0, True)
                quiver_count = sum(
                    isinstance(item, matplotlib.quiver.Quiver)
                    for item in ax.collections
                )
                case.assertEqual(quiver_count, expected_quiver)
                # Contour rendering creates contour-related LineCollections;
                # ensure it appears only in the contour variant.
                line_collection_count = sum(
                    isinstance(item, matplotlib.collections.LineCollection)
                    for item in ax.collections
                )
                case.assertEqual(
                    line_collection_count > 0, bool(expected_contour)
                )
                case.assertIn("Min: f(", ax.get_title())
                case.assertEqual(ax.get_xlabel(), "x")
                case.assertEqual(ax.get_ylabel(), "y")
                case.assertEqual(tuple(np.round(ax.get_xlim(), 5)), (-2, 4))
                case.assertEqual(tuple(np.round(ax.get_ylim(), 5)), (-1, 5))
                plt.close("all")

    def test_zero_gradient_remains_zero(case):
        problem = example_problem()
        problem.objective = sym.Integer(7)
        problem.restrictions = []
        with patch.object(plt, "show"):
            problem.field(resolution=10)
        ax = plt.gcf().axes[0]
        case.assertTrue(np.allclose(np.asarray(ax.images[0].get_array()), 1.0))
        case.assertIn("7.00", ax.get_title())

    def test_failed_feasibility_still_reports_no_sampled_solution(case):
        problem = example_problem()
        x, y = problem.symbols()
        problem.restrictions = [sym.Eq(x + y, sym.Rational(6, 5))]
        with patch.object(plt, "show"):
            with case.assertRaisesRegex(ValueError, "No feasible finite"):
                problem.plot(resolution=2)


if __name__ == "__main__":
    unittest.main()
