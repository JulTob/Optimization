%%writefile optimization_utils.py
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
