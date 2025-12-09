"""
Optimization Modeling and Analysis Library
"""

from .core import Decision, Optimization
from .optimization import (
    analytical_optima,
    integer_optima,
    feasible,
    enumerate_vertices,
    classify_point
)
from .visualization import sunplot
from .interactive import set_sliders

__all__ = [
    # Core classes
    'Decision',
    'Optimization',
    # Optimization functions
    'analytical_optima',
    'integer_optima',
    'feasible',
    'enumerate_vertices',
    'classify_point',
    # Visualization
    'sunplot',
    # Interactive
    'set_sliders',
]

__version__ = '1.0.0'
