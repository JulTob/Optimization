"""Exercise Wizard's public plotting API, not just Graphikon internals."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from Wizard import Decision, Optimization
from Wizard import visualization


def make_c2_fruit_problem(symbolic_call=True):
    """C2 manzanas/platanos example, with both old and new relation syntax."""
    apples = Decision("manzanas", lower=0, upper=6)
    bananas = Decision("platanos", lower=0, upper=7)
    problem = Optimization([apples, bananas])
    if symbolic_call:
        problem[2*apples() + bananas() <= 10]
        problem[apples() + bananas() <= 6]
        objective = (apples()-2)**2 + (bananas()-3)**2
    else:
        problem[2*apples + bananas <= 10]
        problem[apples + bananas <= 6]
        objective = (apples-2)**2 + (bananas-3)**2
    problem.set_objetivo(objective, kind="min")
    return problem


class VisualizationFacadeTests(unittest.TestCase):
    def tearDown(case):
        plt.close("all")

    def test_facade_delegates_arguments(case):
        model = object()
        for name, public_function in (
            ("plot", visualization.plot),
            ("field", visualization.field),
            ("field_contour", visualization.field_contour),
        ):
            with case.subTest(renderer=name):
                with patch(f"Wizard.Graphikon.plots.{name}", return_value="ok") as renderer:
                    result = public_function(
                        model, resolution=34, palette="magma", numeric="Real"
                    )
                renderer.assert_called_once_with(model, 34, "magma", "Real")
                case.assertEqual(result, "ok")

    def test_optimization_methods_use_public_facade(case):
        model = make_c2_fruit_problem()
        for name in ("plot", "field", "field_contour"):
            with case.subTest(method=name):
                with patch(f"Wizard.visualization.{name}", return_value="ok") as renderer:
                    result = getattr(model, name)(resolution=17, palette="inferno")
                renderer.assert_called_once_with(model, 17, "inferno", "Real")
                case.assertEqual(result, "ok")

    def test_real_c2_model_renders_from_both_interfaces(case):
        for symbolic_call in (True, False):
            model = make_c2_fruit_problem(symbolic_call)
            for method in ("plot", "field", "field_contour"):
                with case.subTest(legacy_syntax=symbolic_call, method=method):
                    with patch.object(plt, "show"):
                        getattr(model, method)(resolution=18, palette="magma")
                    ax = plt.gcf().axes[0]
                    case.assertEqual(ax.get_xlabel(), "manzanas")
                    case.assertEqual(ax.get_ylabel(), "platanos")
                    case.assertEqual(tuple(np.round(ax.get_xlim(), 6)), (0, 6))
                    case.assertEqual(tuple(np.round(ax.get_ylim(), 6)), (0, 7))
                    case.assertIn("Min: f(", ax.get_title())
                    background = np.asarray(ax.images[0].get_array())
                    case.assertEqual(background.shape, (18, 18, 3))
                    plt.close("all")

    def test_nonzero_lower_bounds_preserved(case):
        x = Decision("x", lower=-5, upper=4)
        y = Decision("y", lower=-3, upper=2)
        model = Optimization([x, y])
        model[x + y <= 1]
        model.set_objetivo((x+2)**2 + (y+1)**2, kind="min")
        with patch.object(plt, "show"):
            model.plot(resolution=15, palette="inferno")
        ax = plt.gcf().axes[0]
        case.assertEqual(tuple(np.round(ax.get_xlim(), 6)), (-5, 4))
        case.assertEqual(tuple(np.round(ax.get_ylim(), 6)), (-3, 2))


if __name__ == "__main__":
    unittest.main()
