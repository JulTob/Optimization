"""Estimación del ruido y umbral de mejora: la base de los criterios de parada.

¿Cuándo deja de merecer la pena añadir una rampa? Cuando su mejora es
la que se conseguiría ajustando ruido puro. Un grado de libertad
ajustado a ruido N(0, σ²) roba en torno a χ²₁·σ² de suma de cuadrados;
el umbral exige superar el cuantil `confianza` (estándar: 95%) de ese
robo:

    umbral = χ²₁(confianza) · σ̂²      (con χ²₁(0.95) ≈ 3.84)

σ̂² se estima UNA SOLA VEZ, antes de la búsqueda, con diferencias entre
vecinos: dos observaciones casi iguales en x comparten casi la misma
señal, así que su diferencia es casi puro ruido. El umbral depende así
de los datos (de su ruido), no de una constante elegida a mano.
"""
import numpy as np
from scipy.stats import chi2


def varianza_del_ruido_1d(x, y):
    """Estimador de diferencias sucesivas (Rice): media de (Δy)²/2, con x ordenado.

    Al ordenar por x, cada diferencia y_{i+1} − y_i lleva casi nada de
    señal y la resta de dos ruidos independientes, de varianza 2σ²:
    de ahí el medio.
    """
    x, y = _vectores(x, y)
    orden = np.argsort(x)
    return float(np.mean(np.diff(y[orden]) ** 2) / 2.0)


def varianza_del_ruido_2d(Z, y):
    """Lo mismo con el vecino más cercano en el plano: media de (y_i − y_vecino)²/2."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim != 2 or len(Z) != len(y) or y.ndim != 1 or len(y) < 2:
        raise ValueError("Z debe ser (n, d) e y un vector (n,), con n >= 2.")
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(y))):
        raise ValueError("Z e y deben ser finitos.")

    distancias = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.inf)
    vecino = np.argmin(distancias, axis=1)
    return float(np.mean((y - y[vecino]) ** 2) / 2.0)


def umbral_de_mejora(varianza, confianza=0.95):
    """Mejora mínima (en suma de cuadrados) para superar al ruido: χ²₁(confianza)·σ̂²."""
    confianza = float(confianza)
    if not np.isfinite(confianza) or not 0.0 < confianza < 1.0:
        raise ValueError("confianza debe estar entre 0 y 1 (abiertos).")
    varianza = float(varianza)
    if not np.isfinite(varianza) or varianza < 0:
        raise ValueError("La varianza debe ser finita y no negativa.")
    return float(chi2.ppf(confianza, df=1) * varianza)


def _vectores(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("x e y deben ser vectores de igual longitud, con n >= 2.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x e y deben ser finitos.")
    return x, y
