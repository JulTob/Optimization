"""Public visualization facade for Wizard.

Graphikon owns the graphical implementation. This module preserves the
longstanding Wizard visualization entry points without duplicating rendering
or numerical evaluation. Each call owns its temporary plot data.
"""


def plot(problem, resolution=200, palette=None, numeric="Real"):
    """Display a sampled 2D objective and its feasible region."""
    from .Graphikon.plots import plot as render_plot
    return render_plot(problem, resolution, palette, numeric)


def field(problem, resolution=200, palette=None, numeric="Real"):
    """Display a sampled 2D gradient field and feasible region."""
    from .Graphikon.plots import field as render_field
    return render_field(problem, resolution, palette, numeric)


def field_contour(problem, resolution=200, palette=None, numeric="Real"):
    """Display a sampled 2D gradient field with objective contours."""
    from .Graphikon.plots import field_contour as render_contour
    return render_contour(problem, resolution, palette, numeric)
