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

    def set_upper(variable, value):
        """Narrow the variable domain; this is not a model constraint."""
        if value < variable.lower:
            raise ValueError(f"{variable.name} is unfeasible")
        variable.upper = min(value, variable.upper)
        return variable

    def set_lower(variable, value):
        """Narrow the variable domain; this is not a model constraint."""
        if value > variable.upper:
            raise ValueError(f"{variable.name} is unfeasible")
        variable.lower = max(value, variable.lower)
        return variable

    def _sympy_(variable):
        """Expose the underlying symbolic identity to SymPy."""
        return variable.symbol

    # Arithmetic yields SymPy expressions; Decision retains its own domain.
    def __add__(variable, other):
        return variable.symbol + sym.sympify(other)

    def __radd__(variable, other):
        return sym.sympify(other) + variable.symbol

    def __sub__(variable, other):
        return variable.symbol - sym.sympify(other)

    def __rsub__(variable, other):
        return sym.sympify(other) - variable.symbol

    def __mul__(variable, other):
        return variable.symbol * sym.sympify(other)

    def __rmul__(variable, other):
        return sym.sympify(other) * variable.symbol

    def __truediv__(variable, other):
        return variable.symbol / sym.sympify(other)

    def __rtruediv__(variable, other):
        return sym.sympify(other) / variable.symbol

    def __pow__(variable, other):
        return variable.symbol ** sym.sympify(other)

    def __rpow__(variable, other):
        return sym.sympify(other) ** variable.symbol

    def __neg__(variable):
        return -variable.symbol

    def __pos__(variable):
        return +variable.symbol

    # Relational operators build constraints; they never mutate a domain.
    def __le__(variable, other):
        return sym.Le(variable.symbol, sym.sympify(other))

    def __ge__(variable, other):
        return sym.Ge(variable.symbol, sym.sympify(other))

    def __lt__(variable, other):
        return sym.Lt(variable.symbol, sym.sympify(other))

    def __gt__(variable, other):
        return sym.Gt(variable.symbol, sym.sympify(other))

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

    def add_restriction(problem, restriction):
        """Record a symbolic relation; never mistake Python booleans for one."""
        if not isinstance(restriction, sym.Rel):
            raise TypeError(
                "A constraint must be a SymPy relation (<=, >=, <, > or "
                "sym.Eq); Python booleans and expressions are not constraints."
            )
        problem.restrictions.append(restriction)

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
