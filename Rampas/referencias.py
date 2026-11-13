"""Modelos paramétricos de referencia.

Los comparadores del enunciado (recta, polinomios de grado 2 y 3,
potencia) y una logística. Cada función de ajuste recibe (x, y) y
devuelve un predictor: una función que evalúa el modelo ajustado en
puntos nuevos. Es el mismo contrato que las familias de rampas, así
que los experimentos tratan a todos los modelos por igual.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit


def _datos(x, y):
    """Valida y devuelve (x, y) como vectores float."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError(
            "x e y deben ser vectores de igual longitud, con al menos 2 datos."
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x e y deben ser finitos.")
    return x, y


def ajustar_polinomio(x, y, grado):
    """Polinomio de grado dado, por mínimos cuadrados."""
    x, y = _datos(x, y)
    if not isinstance(grado, (int, np.integer)) or grado < 1:
        raise ValueError("grado debe ser un entero mayor o igual que 1.")
    if x.size <= grado:
        raise ValueError("Hacen falta más datos que el grado del polinomio.")

    coeficientes = np.polyfit(x, y, grado)

    def predecir(puntos):
        return np.polyval(coeficientes, np.asarray(puntos, dtype=float))

    return predecir


def ajustar_potencia(x, y):
    """y = a·x^b, por mínimos cuadrados no lineales; exige x > 0."""
    x, y = _datos(x, y)
    if np.any(x <= 0):
        raise ValueError("El modelo potencial necesita x > 0.")

    def formula(t, a, b):
        return a * np.power(t, b)

    parametros, _ = curve_fit(formula, x, y, p0=(1.0, 0.5), maxfev=20000)

    def predecir(puntos):
        return formula(np.asarray(puntos, dtype=float), *parametros)

    return predecir


def ajustar_logistica(x, y):
    """y = c + L·σ(k·(x − x₀)), con iniciales tomadas de los datos."""
    x, y = _datos(x, y)
    ancho = float(np.ptp(x))
    if ancho == 0:
        raise ValueError("El modelo logístico necesita valores de x distintos.")

    def formula(t, nivel, pendiente, centro, base):
        return base + nivel * expit(pendiente * (t - centro))

    iniciales = (
        max(float(np.ptp(y)), 1e-6),
        4.0 / ancho,
        float(np.median(x)),
        float(np.min(y)),
    )
    parametros, _ = curve_fit(formula, x, y, p0=iniciales, maxfev=20000)

    def predecir(puntos):
        return formula(np.asarray(puntos, dtype=float), *parametros)

    return predecir


def catalogo_de_referencias():
    """Los comparadores, como funciones de ajuste (x, y) → predictor."""
    return {
        "Recta": lambda x, y: ajustar_polinomio(x, y, grado=1),
        "Polinomio de grado 2": lambda x, y: ajustar_polinomio(x, y, grado=2),
        "Polinomio de grado 3": lambda x, y: ajustar_polinomio(x, y, grado=3),
        "Potencia": ajustar_potencia,
        "Logística": ajustar_logistica,
    }
