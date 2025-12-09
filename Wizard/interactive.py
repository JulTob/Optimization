"""
Interactive visualization tools for optimization problems.
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import ipywidgets as widgets
from ipywidgets import interact

from .optimization import feasible


def sunplot(P, slider_max=10):
    """
    Interactive sunplot visualization of optimization problem.
    
    Parameters
    ----------
    P :  Optimization
        Optimization problem
    slider_max : float, optional
        Maximum value for all sliders (default 10)
    """
    sliders = set_sliders(P, max=slider_max)
    xs = P.symbols()
    labels = [str(v. symbol) for v in P.variables]
    n = len(labels)

    # Radar geometry
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()

    def plot_frame(**kwargs):
        vals = np.array([kwargs[str(v.symbol)] for v in P.variables])
        current_point = {sym.Symbol(k): v for k, v in kwargs.items()}
        is_feasible = feasible(P, current_point)
        polygon_color = 'lightgreen' if is_feasible else 'tomato'
        polygon_alpha = 0.25 if is_feasible else 0.4

        f_val = float(P.objective.subs({sym:  kwargs[str(sym)] for sym in xs}))

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        overall_max = 0
        for slider_key, slider_widget in sliders.items():
            if hasattr(slider_widget, 'max'):
                overall_max = max(overall_max, slider_widget.max)

        ax.set_rlim(0, overall_max)

        vals_closed = np.concatenate((vals, [vals[0]]))
        angles_closed = angles + [angles[0]]

        ax.plot(angles_closed, vals_closed, linewidth=2, color=polygon_color)
        ax.fill(angles_closed, vals_closed, alpha=polygon_alpha, color=polygon_color)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        radius = np.log(1 + abs(f_val))
        color = plt.cm.inferno(0.5 + np.tanh(f_val / (overall_max + 1)**n))

        grad = [sym.diff(P. objective, v. symbol) for v in P.variables]
        grad_val = np.array([float(g.subs(current_point)) for g in grad])
        for k, gk in enumerate(grad_val):
            print(gk)
            angle = angles[k]
            color_arr = plt.cm.seismic(0.5 - np.tanh(gk / overall_max))
            ax.annotate(
                "",
                xy=(angle, gk),
                xytext=(angle, 0),
                arrowprops=dict(arrowstyle="->", color=color_arr, linewidth=2)
            )

        sun = Circle((0, 0), radius=radius, transform=ax.transData._b, color=color, alpha=0.95)
        ax.add_artist(sun)

        ax.set_title(f"f(x) = {f_val:.3f} {'(Infeasible)' if not is_feasible else ''}", fontsize=14)
        plt.show()

    interact(plot_frame, **sliders)


def set_sliders(P, max=10.0, step=0.1, initial=0.0):
    """
    Build FloatSliders for all decision variables. 

    Parameters
    ----------
    P : Optimization
        Optimization problem
    max : float, optional
        Upper bound for all sliders (default 10.0)
    step : float, optional
        Slider resolution (default 0.1)
    initial : float, optional
        Initial value for all sliders (default 0.0)

    Returns
    -------
    dict[str, widgets.FloatSlider]
        Ready to be passed into sunplot or interact
    """
    sliders = {}

    for v in P.variables:
        if v.numeric == "Integer":
            step = 1
        sliders[str(v.symbol)] = widgets.FloatSlider(
            value=initial,
            min=0. 0,
            max=max,
            step=step,
            description=str(v.symbol),
            continuous_update=True,
            readout_format=".2f"
        )

    return sliders
