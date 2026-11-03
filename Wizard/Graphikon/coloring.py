"""Visual color encoding of satisfied and violated constraint masks."""

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compute_constraint_background(X, masks):
    """Build HSV colors for violated, not satisfied, constraints."""
    colors_rgb = np.array([
        [0.0, 1.0, 0.0],  # Green
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ], dtype=float)
    colors_hsv = rgb_to_hsv(colors_rgb)
    shape = np.shape(X)
    sum_x = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)

    for index, satisfied in enumerate(masks):
        violated = ~satisfied
        hue = colors_hsv[index % len(colors_hsv), 0]
        angle = 2.0 * np.pi * hue
        sum_x[violated] += np.cos(angle)
        sum_y[violated] += np.sin(angle)
        count[violated] += 1.0
    return _angles_to_hsv(sum_x, sum_y, count, shape)


def _angles_to_hsv(sum_x, sum_y, count, shape):
    """Convert aggregated color directions into HSV channel arrays."""
    hue = np.zeros(shape, dtype=float)
    saturation = np.zeros(shape, dtype=float)
    brightness = np.zeros(shape, dtype=float)

    active = count > 0
    angles = np.arctan2(sum_y[active], sum_x[active])
    hue[active] = (angles % (2.0 * np.pi)) / (2.0 * np.pi)
    saturation[active] = 1.0
    brightness[active] = np.clip(
        0.4 + 0.15 * (count[active] - 1.0), 0.3, 0.8
    )
    return hue, saturation, brightness


def apply_feasible_coloring(h_bg, s_bg, v_bg, feasible_mask):
    """Display feasible samples in white without modifying input arrays."""
    hue, saturation, brightness = h_bg.copy(), s_bg.copy(), v_bg.copy()
    saturation[feasible_mask] = 0.0
    brightness[feasible_mask] = 1.0
    return hue, saturation, brightness


def convert_hsv_to_rgb(h_bg, s_bg, v_bg):
    """Combine hue, saturation and brightness into one RGB image."""
    return hsv_to_rgb(np.stack([h_bg, s_bg, v_bg], axis=-1))
