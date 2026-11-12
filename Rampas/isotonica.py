"""Regresión isotónica creciente en una variable.

Problema: minimizar Σ (θ_i − y_i)²  con  θ ordenado como x.
Solución única (objetivo estrictamente convexo, restricciones convexas).
Se resuelve con PAVA: promediar bloques vecinos que rompen el orden.
"""
import numpy as np
from dataclasses import dataclass
from .senales import matriz_de_rampas


@dataclass(frozen=True)
class Nudos:
    """Posiciones distintas de x y el nivel isotónico en cada una."""
    posiciones: np.ndarray
    niveles: np.ndarray


def agrupar_empates(x, y):
    """Un nudo por valor distinto de x. Peso = cuántas observaciones lo forman."""
    posiciones, indice, pesos = np.unique(
            x,
            return_inverse=True,
            return_counts=True,
            )
    sumas = np.bincount(indice, weights=y)
    return posiciones, sumas / pesos, pesos.astype(float)


def pava(valores, pesos):
    """Niveles crecientes más cercanos a `valores`, ponderados por `pesos`."""
    medias, masas, longitudes = [], [], []
    for valor, peso in zip(valores, pesos):
        medias.append(valor)
        masas.append(peso)
        longitudes.append(1)
        while len(medias) > 1 and medias[-2] > medias[-1]:
            masa = masas[-2] + masas[-1]
            media = (medias[-2] * masas[-2] + medias[-1] * masas[-1]) / masa
            longitud = longitudes[-2] + longitudes[-1]
            del medias[-2:], masas[-2:], longitudes[-2:]
            medias.append(media)
            masas.append(masa)
            longitudes.append(longitud)
    return np.repeat(medias, longitudes)


def ajustar_isotonica(x, y):
    """Isotónica creciente de y sobre x. Devuelve los nudos."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("x e y deben ser vectores no vacíos de igual longitud.")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x e y deben ser finitos.")
    posiciones, medias, pesos = agrupar_empates(x, y)
    return Nudos(posiciones, pava(medias, pesos))


def a_rampas(nudos):
    """Rampas que reproducen la interpolación lineal de los nudos (§2.3).

    nivel_0 + Σ (nivel_j − nivel_{j−1}) · R_{t_{j−1}, t_j}
    Devuelve (base, inicios, anchos, pesos); los pesos son ≥ 0.
    """
    t, v = nudos.posiciones, nudos.niveles
    return v[0], t[:-1], np.diff(t), np.diff(v)


def evaluar_nudos(nudos, x):
    """Valor de las rampas de los nudos en x. Constante fuera del rango."""
    base, inicios, anchos, pesos = a_rampas(nudos)
    return base + matriz_de_rampas(x, inicios, anchos) @ pesos


def evaluar_escalones(nudos, x):
    """Extensión por escalones: el nivel del último nudo ≤ x. Constante fuera del rango."""
    indice = np.searchsorted(nudos.posiciones, x, side="right") - 1
    return nudos.niveles[np.clip(indice, 0, len(nudos.niveles) - 1)]
