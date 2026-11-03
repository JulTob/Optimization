"""Smoke-test the existing Graphikon renderers without displaying windows."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from Wizard.Graphikon import plots


def sample_problem():
    x, y = sym.symbols("x y")
    return SimpleNamespace(
        variables=[
            SimpleNamespace(name="x", lower=0, upper=5, numeric="Real"),
            SimpleNamespace(name="y", lower=0, upper=5, numeric="Real"),
        ],
        symbols=lambda: [x, y],
        objective=(x - 2)**2 + (y - 3)**2,
        kind="min",
        restrictions=[sym.Le(x + y, 6)],
    )


class GraphikonPlotTests(unittest.TestCase):
    def tearDown(case):
        plt.close("all")

    def test_all_plot_types(case):
        problem = sample_problem()
        for renderer in (plots.plot, plots.field, plots.field_contour):
            with case.subTest(renderer=renderer.__name__):
                with patch.object(plt, "show"):
                    renderer(problem, resolution=18, palette="magma")
                axes = plt.gcf().axes[0]
                case.assertEqual(
                    (axes.get_xlabel(), axes.get_ylabel()), ("x", "y")
                )
                case.assertIn("Min: f(", axes.get_title())
                image = np.asarray(axes.images[0].get_array())
                case.assertEqual(image.shape, (18, 18, 3))
                case.assertEqual(image.dtype.kind, "f")
                plt.close("all")

    def test_constant_objective_without_constraints(case):
        problem = sample_problem()
        problem.objective = sym.Integer(4)
        problem.restrictions = []
        for renderer in (plots.plot, plots.field):
            with case.subTest(renderer=renderer.__name__):
                with patch.object(plt, "show"):
                    renderer(problem, resolution=12, palette="inferno")
                axes = plt.gcf().axes[0]
                case.assertTrue(np.allclose(axes.images[0].get_array(), 1.))
                case.assertIn("4.00", axes.get_title())
                plt.close("all")

    def test_unsampled_equality_does_not_create_false_optimum(case):
        problem = sample_problem()
        x, y = problem.symbols()
        problem.restrictions = [sym.Eq(x + y, sym.Rational(6, 5))]
        with patch.object(plt, "show"):
            with case.assertRaisesRegex(ValueError, "No feasible finite"):
                plots.plot(problem, resolution=2, palette="magma")


if __name__ == "__main__":
    unittest.main()
