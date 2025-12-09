"""
Core classes for optimization modeling. 
"""

import numpy as np
import sympy as sym
from matplotlib. colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt


class Decision:
    """Represents a decision variable in an optimization problem."""
    
    def __init__(self, 
                 symbol, 
                 value=None,
                 lower=0,
                 upper=np.inf,
                 numeric="Real"):
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
        return self <= value

    def set_lower(self, value):
        return self >= value

    def __le__(self, value):
        if value < self.lower:
            raise ValueError(f"{self.name} is unfeasible")
        self. upper = min(value, self.upper)
        return self

    def __ge__(self, value):
        if value > self.upper:
            raise ValueError(f"{self.name} is unfeasible")
        self.lower = max(value, self.lower)
        return self

    def domain(self):
        return (self.lower, self.upper)

    def __abs__(self):
        return abs(self. upper - self.lower)

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
            self.value = float(self. value)
        return self

    def Binary(self):
        self.numeric = "Integer"
        self.lower = 0
        self.upper = 1
        return self


class Optimization:
    """Main class for defining and solving optimization problems."""
    
    def __init__(self, variables, function=None, kind="min"):
        self.variables = variables
        self.objective = None
        self.kind = None
        self.restrictions = []

    def symbols(self):
        return [v.symbol for v in self. variables]

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
        X = ", ".join([str(x. name) for x in self.variables])
        rest = "\n\t". join([str(x) for x in self.restrictions])
        return f"""f({X}) = {self.objective}
        {self.kind}imization:  {vars}
        s.t. 
        {rest}"""

    def plot(self, resolution=200, palette=None, numeric="Real"):
        """Scatter plot visualization of the optimization problem."""
        from . visualization import plot
        return plot(self, resolution, palette, numeric)

    def field(self, resolution=200, palette=None, numeric="Real"):
        """Vector field visualization of the optimization problem."""
        from . visualization import field
        return field(self, resolution, palette, numeric)

    def field_contour(self, resolution=200, palette=None, numeric="Real"):
        """Contour plot with vector field visualization."""
        from .visualization import field_contour
        return field_contour(self, resolution, palette, numeric)
