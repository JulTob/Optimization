"""Regression tests for Wizard's symbolic decision-variable DSL.

C2 notebook compatibility: x() remains a SymPy symbol, explicit set_upper/
set_lower still narrow the domain, and P[relation] still stores relations.
"""
import unittest

import sympy as sym

from Wizard import Decision, Optimization


class DecisionAlgebraTests(unittest.TestCase):
    def setUp(case):
        case.x = Decision("x", lower=0, upper=10)
        case.y = Decision("y", lower=-2, upper=8)

    def test_arithmetic_expressions(case):
        x, y = case.x, case.y
        expressions = (
            (x + y, x() + y()),
            (5 + x, 5 + x()),
            (x - y, x() - y()),
            (5 - x, 5 - x()),
            (x * y, x() * y()),
            (3 * x, 3 * x()),
            (x / (y + 3), x() / (y() + 3)),
            (8 / x, 8 / x()),
            (x ** 2, x() ** 2),
            (2 ** x, 2 ** x()),
            (-x, -x()),
            (+x, +x()),
            (sum([x, y]), x() + y()),
        )
        for actual, expected in expressions:
            with case.subTest(actual=actual):
                case.assertEqual(sym.simplify(actual - expected), 0)

    def test_native_sympy_interoperability(case):
        x, y = case.x, case.y
        case.assertEqual(sym.sympify(x), x())
        case.assertEqual(sym.expand((x + y)**2), sym.expand((x() + y())**2))
        case.assertEqual(sym.diff(x + y**2, y()), 2*y())
        case.assertEqual(sym.Eq(x + y, 6), sym.Eq(x() + y(), 6))

    def test_relations_are_not_domain_mutations(case):
        x, y = case.x, case.y
        before = (x.domain(), y.domain())
        relations = (
            (x <= 6, sym.Le(x(), 6)),
            (x >= -1, sym.Ge(x(), -1)),
            (x < y, sym.Lt(x(), y())),
            (x > y, sym.Gt(x(), y())),
        )
        for actual, expected in relations:
            case.assertEqual(actual, expected)
            case.assertIsInstance(actual, sym.Rel)
        case.assertEqual((x.domain(), y.domain()), before)

    def test_explicit_domain_bounds_remain_compatible(case):
        x = case.x
        case.assertIs(x.set_upper(6), x)
        case.assertEqual(x.domain(), (0, 6))
        case.assertIs(x.set_lower(2), x)
        case.assertEqual(x.domain(), (2, 6))
        case.assertTrue(x.in_domain(3))
        case.assertFalse(x.in_domain(7))
        with case.assertRaisesRegex(ValueError, "unfeasible"):
            x.set_upper(1)
        case.assertEqual(x.domain(), (2, 6))

    def test_problem_adds_symbolic_relations(case):
        x, y = case.x, case.y
        problem = Optimization([x, y])
        problem[x + y <= 6]
        problem[2*x - y >= 1]
        problem[sym.Eq(x + y, 5)]
        problem[x < 7]
        problem[y > -3]
        case.assertEqual(len(problem.restrictions), 5)
        case.assertTrue(all(isinstance(r, sym.Rel) for r in problem.restrictions))
        case.assertEqual(problem.symbols(), [x(), y()])

    def test_reject_boolean_and_expression_constraints(case):
        problem = Optimization([case.x, case.y])
        for invalid in (True, False, sym.true, sym.false, case.x + case.y, 42):
            with case.subTest(invalid=invalid):
                with case.assertRaisesRegex(TypeError, "SymPy relation"):
                    problem.add_restriction(invalid)
        case.assertFalse(problem.restrictions)

    def test_c2_notebook_legacy_call_syntax(case):
        apples = Decision("manzanas")
        bananas = Decision("platanos")
        problem = Optimization([apples, bananas])
        problem[2*apples() + bananas() <= 10]
        problem[apples() + bananas() <= 6]
        problem.set_objetivo(
            (apples()-2)**2 + (bananas()-3)**2, kind="min"
        )
        apples.set_upper(6)
        bananas.set_upper(7)
        case.assertEqual(problem.kind, "min")
        case.assertEqual(problem.variables[0].domain(), (0, 6))
        case.assertEqual(problem.variables[1].domain(), (0, 7))
        case.assertEqual(len(problem.restrictions), 2)
        case.assertEqual(
            sym.simplify(problem.objective-((apples()-2)**2+(bananas()-3)**2)), 0
        )

    def test_c3_binary_and_sum_notation(case):
        choices = [Decision(f"city_{i}").Binary() for i in range(3)]
        problem = Optimization(choices)
        problem[sum(choices) >= 1]
        problem[sum(choices) <= 2]
        case.assertEqual(len(problem.restrictions), 2)
        case.assertTrue(all(v.domain() == (0, 1) for v in choices))


if __name__ == "__main__":
    unittest.main()
