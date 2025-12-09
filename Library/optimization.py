"""
Optimization algorithms and utility functions.
"""

import numpy as np
import sympy as sym
import itertools


def classify_point(H, point):
    """
    Classify a critical point based on the Hessian matrix.
    
    Parameters
    ----------
    H : sympy.Matrix
        Hessian matrix
    point : dict
        Point to classify
        
    Returns
    -------
    str
        Classification:  'min', 'max', 'saddle', or 'degenerate'
    """
    H_eval = H.subs(point)
    eigenvals = H_eval.eigenvals()
    eigs = list(eigenvals.keys())

    if all(ev > 0 for ev in eigs):
        return "min"
    if all(ev < 0 for ev in eigs):
        return "max"
    if any(ev < 0 for ev in eigs) and any(ev > 0 for ev in eigs):
        return "saddle"
    return "degenerate"


def feasible(P, point):
    """
    Check if a point satisfies all constraints and domain bounds.
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    point : dict
        Point to check
        
    Returns
    -------
    bool
        True if point is feasible, False otherwise
    """
    # Check all restrictions
    for r in P.restrictions:
        try:
            lhs_substituted = r.lhs.subs(point)
            rhs_substituted = r.rhs.subs(point)

            if not (lhs_substituted. is_Number and rhs_substituted.is_Number):
                return False

            lhs_val = float(lhs_substituted)
            rhs_val = float(rhs_substituted)

            if isinstance(r, sym.Le):
                if lhs_val > rhs_val:
                    return False
            elif isinstance(r, sym.Ge):
                if lhs_val < rhs_val:
                    return False
            elif isinstance(r, sym. Eq):
                if not np.isclose(lhs_val, rhs_val):
                    return False
            else: 
                raise NotImplementedError(f"Unsupported SymPy Relational type: {type(r)}")

        except (TypeError, AttributeError, ValueError) as e:
            print(f"Warning: Could not evaluate restriction {r} at point {point}. Error: {e}")
            return False

    # Check domain bounds from Decision objects
    for v_dec in P.variables:
        if v_dec.symbol not in point:
            return False

        val = point[v_dec.symbol]
        if not float(v_dec.lower) <= float(val) <= float(v_dec.upper):
            return False

    return True


def enumerate_vertices(P):
    """
    Enumerate all vertices of the feasible region.
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
        
    Returns
    -------
    list[dict]
        List of feasible vertices
    """
    xs = P.symbols()
    n = len(xs)
    equations = [r.lhs - r.rhs for r in P.restrictions]
    vertices = []

    for combo in itertools.combinations(equations, n):
        sol = sym.solve(combo, xs, dict=True)
        for s in sol: 
            if all(v in s and s[v]. is_real for v in xs):
                if feasible(P, s):
                    vertices.append(s)

    return vertices


def analytical_optima(OP, steps=True):
    """
    Find optimal solution using analytical methods (gradient and critical points).
    
    Parameters
    ----------
    OP :  Optimization
        Optimization problem
    steps : bool, optional
        Print intermediate steps (default True)
    """
    P = OP
    xs = P.symbols()
    f = P.objective

    # Gradient
    grad = [sym.diff(f, x) for x in xs]

    # Hessian
    H = sym.hessian(f, xs)

    if steps:
        print("\nGradiente & Hessiana:")
        print(grad)
        print(H)

    # Critical points
    critical_points = sym.solve(grad, xs, dict=True)

    if steps:
        print("\nPuntos críticos:")
        print(critical_points)

    candidates = []

    # Evaluate critical points
    for cp in critical_points:
        if feasible(P, cp):
            fval = float(f.subs(cp))
            kind = classify_point(H, cp)
            candidates.append((cp, fval, kind, "critical"))
            if steps:
                print(f"\nCrítico factible: {cp}, f={fval}, tipo={kind}")
        else:
            if steps:
                print(f"\nCrítico NO factible: {cp}")

    vertices = enumerate_vertices(P)

    for v in vertices:
        if feasible(P, v):
            fval = float(f.subs(v))
            candidates.append((v, fval, "vertex", "vertex"))
            if steps:
                print(f"Vértice factible: {v}, f={fval}, tipo=vertex")

    # Select optimum
    if not candidates:
        print("No feasible points found.")
        return

    if P.kind == "max":
        p_opt, f_opt, kind_opt, src_opt = max(candidates, key=lambda t: t[1])
    else:
        p_opt, f_opt, kind_opt, src_opt = min(candidates, key=lambda t:  t[1])

    print(f"\nÓptimo {P.kind}: f={f_opt} en {p_opt}")


def integer_optima(OP):
    """
    Find optimal solution for integer programming problems via enumeration.
    
    Parameters
    ----------
    OP :  Optimization
        Optimization problem with integer variables
        
    Returns
    -------
    tuple
        (best_assignment, best_value)
    """
    variables = OP.variables

    # Build domains
    domains = []
    for x in variables:
        low = x.lower
        high = x.upper

        if x.numeric == "Integer" and high == np.inf:
            raise ValueError(
                f"Dominio infinito no permitido para variable entera:  {x. name}.  "
                "Debe establecer un límite superior finito para variables enteras."
            )

        if x.numeric == "Integer": 
            low = int(low)
            high = int(high)

        domains.append(range(low, high + 1))

    best_value = None
    best_assignment = None

    # Enumerate combinations
    for values in itertools.product(*domains):
        assignment = dict(zip([x() for x in variables], values))

        if not feasible(OP, assignment):
            continue

        # Evaluate objective
        value = OP.objective.subs(assignment)

        if best_value is None: 
            best_value = value
            best_assignment = assignment
        else:
            if OP.kind == "max" and value > best_value: 
                best_value = value
                best_assignment = assignment
            elif OP.kind == "min" and value < best_value: 
                best_value = value
                best_assignment = assignment

    return best_assignment, best_value
