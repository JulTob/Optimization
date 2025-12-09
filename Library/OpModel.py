import numpy as np
import sympy as sym
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt

class Decision:
    def __init__(self, 
                 symbol, 
                 value= None,
                 lower=0,
                 upper=np.inf,
                 numeric = "Real" ):
        self.name = symbol
        self.symbol = sym.Symbol(symbol)
        self.lower = lower
        self.upper = upper
        self.value = value
        self.numeric = numeric

    def __call__(self):
        return self.symbol

    def __contains__(self, x):
        return self.lower <= x <= self.upper

    def in_domain(self, x):
        return x in self

    def __str__(self):
        if self.value is not None:
            return f"{self.name} ∈ [{self.lower}, {self.upper}] := {self.value}"
        else:
            return f"{self.name} ∈ [{self.lower}, {self.upper}]"

    def set_upper(self, value):
        self <= value

    def set_lower(self, value):
        self >= value

    def __le__(self, value):
        if value < self.lower:
            raise ValueError(f"{self.name} is unfeasible")
        self.upper = min(value, self.upper)
        return self

    def __ge__(self, value):
        if value > self.upper:
                raise ValueError(f"{self.name} is unfeasible")
        self.lower = max(value, self.lower)
        return self

    def domain(self):
        return (self.lower, self.upper)

    def __abs__(self):
        return abs(self.upper - self.lower)

    def __len__(self):
        return self.upper - self.lower

    def Integer(self):
        self.numeric = "Integer"
        if self.value is not None:
            self.value = int(self.value)
        return self

    def Real(self):
        self.numeric = "Real"
        if self.value is not None:
            self.value = float(self.value)
        return self

    def Binary(self):
        self.numeric = "Integer"
        self.lower = 0
        self.upper = 1
        return self


class Optimization:
    def __init__(self, variables, function=None, kind = "min"):
        self.variables = variables
        self.objective = None
        self.kind = None
        self.restrictions = []

    def symbols(self):
        return [v.symbol for v in self.variables]

    def set_maximize(self):
        self.kind = "max"

    def set_minimize(self):
        self.kind = "min"

    def __call__(self, f):
        self.objective = f
        return self

    def set_objetivo(self, function, kind="min"):
        self.objective = function
        self.kind = kind

    def __pos__(self):
        self.set_maximize()
        return self

    def __neg__(self):
        self.set_minimize()
        return self

    def add_restriction(self, restriction):
        self.restrictions.append(restriction)

    def __getitem__(self, restriction):
        self.add_restriction(restriction)

    def __str__(self):
        vars = ", ".join([str(x) for x in self.variables])
        X = ", ".join([str(x.name) for x in self.variables])
        rest = "\n\t".join([str(x) for x in self.restrictions])
        return f"""f({X}) = {self.objective}
        {self.kind}imization: {vars}
        s.t.
        {rest}"""


    def plot(self,
             resolution=200,
             palette=None,
             numeric = "Real"):
        dot_alpha = 1
        if palette is None:
            palette = "gray"
            dot_alpha = 0.7

        xs = self.symbols()
        d1, d2 = self.variables
        xmin, xmax = 0, d1.upper
        ymin, ymax = 0, d2.upper

        if d1.numeric == "Real":
            x_vals = np.linspace(xmin, xmax, resolution)
        elif d1.numeric == "Integer":
            x_vals = np.arange(int(xmin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        if d2.numeric == "Real":
            y_vals = np.linspace(ymin, ymax, resolution)
        elif d2.numeric == "Integer":
            y_vals = np.arange(int(ymin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        X, Y = np.meshgrid(x_vals, y_vals)

        if self.objective is None:
            raise ValueError("No objective function defined.")

        f_num = sym.lambdify(xs, self.objective, "numpy")
        Z = f_num(X, Y)

        masks = []
        for r in self.restrictions:
            g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
            G = g_num(X, Y)
            masks.append(G <= 0)

        feasible_mask = np.ones_like(X, dtype=bool)
        for mask in masks:
            feasible_mask &= mask

        fig, ax = plt.subplots(figsize=(8, 8))

        colors_rgb = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])

        colors_hsv = rgb_to_hsv(colors_rgb)

        shape = X.shape
        sum_x = np.zeros(shape, dtype=float)
        sum_y = np.zeros(shape, dtype=float)
        count = np.zeros(shape, dtype=float)

        for i, mask in enumerate(masks):
            h, s, v = colors_hsv[i % len(colors_hsv)]
            angle = 2 * np.pi * h

            cos_h = np.cos(angle)
            sin_h = np.sin(angle)

            sum_x[mask] += cos_h
            sum_y[mask] += sin_h
            count[mask] += 1.0

        h_bg = np.zeros(shape, dtype=float)
        s_bg = np.zeros(shape, dtype=float)
        v_bg = np.zeros(shape, dtype=float)

        active = count > 0

        angles = np.arctan2(sum_y[active], sum_x[active])
        h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)

        s_bg[active] = 1.0

        v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

        s_bg[feasible_mask] = 0.0
        v_bg[feasible_mask] = 1.0

        hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
        rgb_bg = hsv_to_rgb(hsv_bg)

        ax.imshow(
            rgb_bg,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower"
        )

        satisfied_count = sum(mask for mask in masks)
        violated = len(masks) - satisfied_count

        n_points = resolution**2
        base_r = 6e4 / n_points
        R = base_r * satisfied_count / (1 + violated)

        ax.scatter(X.flatten(), Y.flatten(),
                s=R.flatten(),
                c=Z.flatten(),
                cmap=palette,
                edgecolors="none",
                alpha=dot_alpha)

        Z_feasible = np.where(feasible_mask, Z, np.nan)
        if self.kind == "max": opt = np.nanargmax(Z_feasible)
        else: opt = np.nanargmin(Z_feasible)
        x_opt = X.flatten()[opt]
        y_opt = Y.flatten()[opt]
        f_opt = Z.flatten()[opt]

        ax.scatter([x_opt], [y_opt],
                color="gold",
                edgecolor="black",
                s=220, zorder=5)

        opt_mask = (Z_feasible == f_opt)
        x_opt_all = X[opt_mask]
        y_opt_all = Y[opt_mask]

        ax.set_title(f"{self.kind}: ≈ f({x_opt:.2f}, {y_opt:.2f}) = {f_opt:.2f}")
        ax.set_xlabel(d1.name)
        ax.set_ylabel(d2.name)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.scatter(
            x_opt_all, y_opt_all,
            color="gold",
            edgecolor="goldenrod",
            s=40,
            zorder=5
            )

        plt.show()


    def field(self,
              resolution=200,
              palette=None,
              numeric="Real"):
        dot_alpha = 1
        if palette is None:
            palette = "gray"
            dot_alpha = 0.7

        xs = self.symbols()
        d1, d2 = self.variables
        xmin, xmax = 0, d1.upper
        ymin, ymax = 0, d2.upper

        if d1.numeric == "Real":
            x_vals = np.linspace(xmin, xmax, resolution)
        elif d1.numeric == "Integer":
            x_vals = np.arange(int(xmin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        if d2.numeric == "Real":
            y_vals = np.linspace(ymin, ymax, resolution)
        elif d2.numeric == "Integer":
            y_vals = np.arange(int(ymin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        X, Y = np.meshgrid(x_vals, y_vals)

        if self.objective is None:
            raise ValueError("No objective function defined.")

        f_num = sym.lambdify(xs, self.objective, "numpy")
        Z = f_num(X, Y)

        masks = []
        for r in self.restrictions:
            g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
            G = g_num(X, Y)
            masks.append(G <= 0)

        feasible_mask = np.ones_like(X, dtype=bool)
        for mask in masks:
            feasible_mask &= mask

        fig, ax = plt.subplots(figsize=(8, 8))

        colors_rgb = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        colors_hsv = rgb_to_hsv(colors_rgb)

        shape = X.shape
        sum_x = np.zeros(shape, dtype=float)
        sum_y = np.zeros(shape, dtype=float)
        count = np.zeros(shape, dtype=float)

        for i, mask in enumerate(masks):
            h, s, v = colors_hsv[i % len(colors_hsv)]
            angle = 2 * np.pi * h
            cos_h = np.cos(angle)
            sin_h = np.sin(angle)
            sum_x[mask] += cos_h
            sum_y[mask] += sin_h
            count[mask] += 1.0

        h_bg = np.zeros(shape, dtype=float)
        s_bg = np.zeros(shape, dtype=float)
        v_bg = np.zeros(shape, dtype=float)

        active = count > 0
        angles = np.arctan2(sum_y[active], sum_x[active])
        h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)
        s_bg[active] = 1.0
        v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

        s_bg[feasible_mask] = 0.0
        v_bg[feasible_mask] = 1.0

        hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
        rgb_bg = hsv_to_rgb(hsv_bg)

        ax.imshow(
            rgb_bg,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower"
        )

        n_constraints = len(masks)
        n_points = X.size

        if n_constraints > 0:
            satisfied_count = np.zeros_like(X, dtype=float)
            for mask in masks:
                satisfied_count += mask
            ratio = satisfied_count / n_constraints
        else:
            ratio = np.ones_like(X, dtype=float)

        size_factor = 0.3 + 0.7 * ratio

        grad = [sym.diff(self.objective, var) for var in xs]
        gx_num = sym.lambdify(xs, grad[0], "numpy")
        gy_num = sym.lambdify(xs, grad[1], "numpy")

        GX_scalar = gx_num(X, Y)
        GY_scalar = gy_num(X, Y)

        if np.isscalar(GX_scalar):
            GX = np.full_like(X, GX_scalar)
        else:
            GX = GX_scalar

        if np.isscalar(GY_scalar):
            GY = np.full_like(Y, GY_scalar)
        else:
            GY = GY_scalar

        norm = np.hypot(GX, GY)
        norm[norm == 0] = 1.0

        base_len = 0.03 * max(xmax - xmin, ymax - ymin)

        U = GX / norm * base_len * size_factor
        V = GY / norm * base_len * size_factor

        step = max(1, resolution // 25)

        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            U[::step, ::step],
            V[::step, ::step],
            Z[::step, ::step],
            cmap=palette,
            alpha=dot_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004
        )

        Z_feasible = np.where(feasible_mask, Z, np.nan)
        if self.kind == "max":
            opt = np.nanargmax(Z_feasible)
        else:
            opt = np.nanargmin(Z_feasible)

        x_opt = X.flatten()[opt]
        y_opt = Y.flatten()[opt]
        f_opt = Z.flatten()[opt]

        ax.scatter(
            [x_opt], [y_opt],
            color="gold",
            edgecolor="black",
            s=220,
            zorder=5
        )

        opt_mask = (Z_feasible == f_opt)
        x_opt_all = X[opt_mask]
        y_opt_all = Y[opt_mask]

        ax.set_title(f"{self.kind}: ≈ f({x_opt:.2f}, {y_opt:.2f}) = {f_opt:.2f}")
        ax.set_xlabel(d1.name)
        ax.set_ylabel(d2.name)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        ax.scatter(
            x_opt_all, y_opt_all,
            color="gold",
            edgecolor="goldenrod",
            s=40,
            zorder=5
        )

        plt.show()
    def field_contour(self,
              resolution=200,
              palette=None,
              numeric="Real"):
        dot_alpha = 1
        if palette is None:
            palette = "gray"
            dot_alpha = 0.7

        xs = self.symbols()
        d1, d2 = self.variables
        xmin, xmax = 0, d1.upper
        ymin, ymax = 0, d2.upper

        if d1.numeric == "Real":
            x_vals = np.linspace(xmin, xmax, resolution)
        elif d1.numeric == "Integer":
            x_vals = np.arange(int(xmin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        if d2.numeric == "Real":
            y_vals = np.linspace(ymin, ymax, resolution)
        elif d2.numeric == "Integer":
            y_vals = np.arange(int(ymin), int(xmax) + 1)
        else:
            raise ValueError("Numeric type not supported")

        X, Y = np.meshgrid(x_vals, y_vals)

        if self.objective is None:
            raise ValueError("No objective function defined.")

        f_num = sym.lambdify(xs, self.objective, "numpy")
        Z = f_num(X, Y)

        masks = []
        for r in self.restrictions:
            g_num = sym.lambdify(xs, r.lhs - r.rhs, "numpy")
            G = g_num(X, Y)
            masks.append(G <= 0)

        feasible_mask = np.ones_like(X, dtype=bool)
        for mask in masks:
            feasible_mask &= mask

        fig, ax = plt.subplots(figsize=(8, 8))

        colors_rgb = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        colors_hsv = rgb_to_hsv(colors_rgb)

        shape = X.shape
        sum_x = np.zeros(shape, dtype=float)
        sum_y = np.zeros(shape, dtype=float)
        count = np.zeros(shape, dtype=float)

        for i, mask in enumerate(masks):
            h, s, v = colors_hsv[i % len(colors_hsv)]
            angle = 2 * np.pi * h
            cos_h = np.cos(angle)
            sin_h = np.sin(angle)
            sum_x[mask] += cos_h
            sum_y[mask] += sin_h
            count[mask] += 1.0

        h_bg = np.zeros(shape, dtype=float)
        s_bg = np.zeros(shape, dtype=float)
        v_bg = np.zeros(shape, dtype=float)

        active = count > 0
        angles = np.arctan2(sum_y[active], sum_x[active])
        h_bg[active] = (angles % (2 * np.pi)) / (2 * np.pi)
        s_bg[active] = 1.0
        v_bg[active] = np.clip(0.4 + 0.15 * (count[active] - 1), 0.3, 0.8)

        s_bg[feasible_mask] = 0.0
        v_bg[feasible_mask] = 1.0

        hsv_bg = np.stack([h_bg, s_bg, v_bg], axis=-1)
        rgb_bg = hsv_to_rgb(hsv_bg)

        ax.imshow(
            rgb_bg,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower"
        )

        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.5, linewidths=0.8)

        n_constraints = len(masks)
        n_points = X.size

        if n_constraints > 0:
            satisfied_count = np.zeros_like(X, dtype=float)
            for mask in masks:
                satisfied_count += mask
            ratio = satisfied_count / n_constraints
        else:
            ratio = np.ones_like(X, dtype=float)

        size_factor = 0.3 + 0.7 * ratio

        grad = [sym.diff(self.objective, var) for var in xs]
        gx_num = sym.lambdify(xs, grad[0], "numpy")
        gy_num = sym.lambdify(xs, grad[1], "numpy")

        GX_scalar = gx_num(X, Y)
        GY_scalar = gy_num(X, Y)

        if np.isscalar(GX_scalar):
            GX = np.full_like(X, GX_scalar)
        else:
            GX = GX_scalar

        if np.isscalar(GY_scalar):
            GY = np.full_like(Y, GY_scalar)
        else:
            GY = GY_scalar

        norm = np.hypot(GX, GY)
        norm[norm == 0] = 1.0

        base_len = 0.03 * max(xmax - xmin, ymax - ymin)

        U = GX / norm * base_len * size_factor
        V = GY / norm * base_len * size_factor

        step = max(1, resolution // 25)

        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            U[::step, ::step],
            V[::step, ::step],
            Z[::step, ::step],
            cmap=palette,
            alpha=dot_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004
        )

        Z_feasible = np.where(feasible_mask, Z, np.nan)
        if self.kind == "max":
            opt = np.nanargmax(Z_feasible)
        else:
            opt = np.nanargmin(Z_feasible)

        x_opt = X.flatten()[opt]
        y_opt = Y.flatten()[opt]
        f_opt = Z.flatten()[opt]

        ax.scatter(
            [x_opt], [y_opt],
            color="gold",
            edgecolor="black",
            s=220,
            zorder=5
        )

        opt_mask = (Z_feasible == f_opt)
        x_opt_all = X[opt_mask]
        y_opt_all = Y[opt_mask]

        ax.set_title(f"{self.kind}: ≈ f({x_opt:.2f}, {y_opt:.2f}) = {f_opt:.2f}")
        ax.set_xlabel(d1.name)
        ax.set_ylabel(d2.name)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        ax.scatter(
            x_opt_all, y_opt_all,
            color="gold",
            edgecolor="goldenrod",
            s=40,
            zorder=5
        )

        plt.show()

def classify_point(H, point):
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


import numpy as np
import sympy as sym

def feasible(P, point):
    # Check all restrictions
    for r in P.restrictions:
        try:
            # Substitute point into LHS and RHS of the relational expression
            # This might result in an expression if not all symbols are in 'point'
            lhs_substituted = r.lhs.subs(point)
            rhs_substituted = r.rhs.subs(point)

            # Check if substitution resulted in non-numeric SymPy expressions
            # If so, this 'point' cannot fully evaluate this constraint, so it's not well-defined for this check
            if not (lhs_substituted.is_Number and rhs_substituted.is_Number):
                # If either side is still symbolic, it means the point doesn't fully define the expression
                # For our context (checking feasibility at a specific point), this usually means it's not fully defined
                return False

            lhs_val = float(lhs_substituted)
            rhs_val = float(rhs_substituted)

            # Check violation based on the type of SymPy relational operator
            if isinstance(r, sym.Le): # Constraint is of type A <= B
                if lhs_val > rhs_val: # Violated if A > B
                    return False
            elif isinstance(r, sym.Ge): # Constraint is of type A >= B
                if lhs_val < rhs_val: # Violated if A < B
                    return False
            elif isinstance(r, sym.Eq): # Constraint is of type A == B
                # For equality, use a numerical tolerance (np.isclose)
                if not np.isclose(lhs_val, rhs_val):
                    return False
            else:
                # Should not happen if only Le, Ge, Eq are used, but good for robustness
                raise NotImplementedError(f"Unsupported SymPy Relational type: {type(r)}")

        except (TypeError, AttributeError, ValueError) as e:
            # Catch errors during substitution or float conversion for debugging
            print(f"Warning: Could not evaluate restriction {r} at point {point}. Error: {e}")
            return False

    # Check domain bounds from Decision objects
    for v_dec in P.variables:
        # Ensure the current variable is in the point dictionary
        if v_dec.symbol not in point:
            # If a variable is missing from the point, the point is incomplete for feasibility check.
            return False

        val = point[v_dec.symbol]
        # Convert SymPy numeric types to Python float for comparison with v.lower/upper
        # This handles potential SymPy Integers/Floats from substitution
        if not float(v_dec.lower) <= float(val) <= float(v_dec.upper):
            return False

    return True

import itertools
import sympy as sym

def enumerate_vertices(P):
    xs = P.symbols()
    n = len(xs)

    equations = [r.lhs - r.rhs for r in P.restrictions]

    vertices = []

    # take every combination of n equations
    for combo in itertools.combinations(equations, n):
        sol = sym.solve(combo, xs, dict=True)
        for s in sol:
            # keep only numerical complete solutions
            if all(v in s and s[v].is_real for v in xs):
                if feasible(P, s):
                    vertices.append(s)

    return vertices

def analytical_optima(OP, steps=True):
    import sympy as sym
    from itertools import combinations
    P = OP
    xs = P.symbols()
    f = P.objective

    # Gradiente
    grad = [sym.diff(f, x) for x in xs]

    # Hessiano
    H = sym.hessian(f, xs)

    if steps:
        print("\nGradiente & Hessiana:")
        print(grad)
        print(H)

    # ====== PUNTOS CRÍTICOS ======
    critical_points = sym.solve(grad, xs, dict=True)

    if steps:
        print("\nPuntos críticos:")
        print(critical_points)
        print(classify_point(H, critical_points))

    candidates = []

    # Evaluar puntos críticos
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
        if feasible(P,v):
            fval = float(f.subs(v))
            candidates.append((v, fval, "vertex", "vertex"))
            if steps:
                print(f"Vértice factible: {v}, f={fval}, tipo=vertex")

    # ====== SELECT OPTIMUM ======
    if not candidates:
        print("No feasible points found.")
        return

    # candidates = [(point, fval, kind, source), ...]
    if P.kind == "max":
        p_opt, f_opt, kind_opt, src_opt = max(candidates, key=lambda t: t[1])
    else:
        p_opt, f_opt, kind_opt, src_opt = min(candidates, key=lambda t: t[1])

    print(f"\nÓptimo {P.kind}: f={f_opt} en {p_opt}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import ipywidgets as widgets
from ipywidgets import interact # Import interact function
import sympy as sym # Ensure sympy is imported for feasible function


def sunplot(P, slider_max=10):
    sliders = set_sliders(P, max=slider_max)
    xs = P.symbols()
    labels = [str(v.symbol) for v in P.variables]
    n = len(labels)

    # Radar geometry
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()

    def plot_frame(**kwargs):
        # substitute chosen values
        vals = np.array([kwargs[str(v.symbol)] for v in P.variables])

        # Create a point dictionary for the feasible check
        current_point = {sym.Symbol(k): v for k, v in kwargs.items()}

        # Check feasibility of the current point
        is_feasible = feasible(P, current_point) # Assuming 'feasible' function is available in scope
        polygon_color = 'lightgreen' if is_feasible else 'tomato'
        polygon_alpha = 0.25 if is_feasible else 0.4

        # evaluate objective
        f_val = float(P.objective.subs({sym: kwargs[str(sym)] for sym in xs}))

        # === Radar Plot ===
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)

        # Dynamically determine the overall maximum slider value for radar extent
        overall_max = 0
        for slider_key, slider_widget in sliders.items():
            if hasattr(slider_widget, 'max'): # Check if it's a widget with a 'max' attribute
                overall_max = max(overall_max, slider_widget.max)

        # Set radial limits based on overall_max
        ax.set_rlim(0, overall_max)

        # close polygon
        vals_closed = np.concatenate((vals, [vals[0]]))
        angles_closed = angles + [angles[0]]

        # draw radar outline
        ax.plot(angles_closed, vals_closed, linewidth=2, color=polygon_color)
        ax.fill(angles_closed, vals_closed, alpha=polygon_alpha, color=polygon_color)

        # label axes
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)


        # === Sun in center ===
        radius = np.log(1+abs(f_val))
        color = plt.cm.inferno(0.5 + np.tanh(f_val/(overall_max+1)**n))


        # Growth Vectors
        grad = [sym.diff(P.objective, v.symbol) for v in P.variables]
        grad_val = np.array([float(g.subs(current_point)) for g in grad])
        for k, gk in enumerate(grad_val):
            print(gk)
            angle = angles[k]
            color_arr = plt.cm.seismic(0.5 - np.tanh(gk/overall_max))
            ax.annotate(
                "",
                xy=(angle, gk),
                xytext=(angle, 0),
                arrowprops=dict(arrowstyle="->", color=color_arr, linewidth=2)
                )

        sun = Circle((0,0), radius=radius, transform=ax.transData._b, color=color, alpha=0.95)
        ax.add_artist(sun)

        ax.set_title(f"f(x) = {f_val:.3f} {'(Infeasible)' if not is_feasible else ''}", fontsize=14)
        plt.show()

    interact(plot_frame, **sliders)

import ipywidgets as widgets

def set_sliders(P, max=10.0, step=0.1, initial=0.0):
    """
    Build FloatSliders for all decision variables in an Optimization problem.

    Parameters
    ----------
    P : Optimization - Your optimization problem
    max : float - Upper bound for all sliders
    step : float - Slider resolution
    initial : float - Initial value for all sliders (default 0)

    Returns
    -------
    dict[str, widgets.FloatSlider]
        Ready to be passed into sunplot or interact
    """
    sliders = {}

    for v in P.variables:
        if v.numeric == "Integer": step = 1
        sliders[str(v.symbol)] = widgets.FloatSlider(
            value=initial,
            min=0.0,
            max=max,
            step=step,
            description=str(v.symbol),
            continuous_update=True,
            readout_format=".2f"
        )

    return sliders
import itertools
import sympy as sym
import numpy as np # Import numpy for np.inf

def integer_optima(OP):
    variables = OP.variables

    # 1. construir dominios
    domains = []
    for x in variables:
        low = x.lower
        high = x.upper

        # Enforce finite bounds for integer variables
        if x.numeric == "Integer" and high == np.inf:
            raise ValueError(
                f"Dominio infinito no permitido para variable entera: {x.name}. "
                "Debe establecer un límite superior finito para variables enteras."
            )

        # Ensure bounds are integers for range()
        if x.numeric == "Integer":
            low = int(low)
            high = int(high)

        domains.append(range(low, high+1))

    best_value = None
    best_assignment = None

    # 2. enumerar combinaciones
    for values in itertools.product(*domains):
        assignment = dict(zip([x() for x in variables], values))

        # 3. comprobar restricciones
        feasible = True
        for r in OP.restrictions:
            if not bool(r.subs(assignment)):
                feasible = False
                break

        if not feasible:
            continue

        # 4. evaluar objetivo
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
