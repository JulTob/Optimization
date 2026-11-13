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
de los datos (de su ruido), no de una constante elegida a mano, y el
estimador es el mismo en cualquier dimensión.
"""
import numpy as np
from scipy.stats import chi2


def varianza_del_ruido(X, y):
    """σ̂²: media de (y_i − y_vecino)² / 2, con el vecino más cercano en X.

    El ruido se mide SIEMPRE en y; X solo decide quién es vecino de
    quién. Dos observaciones casi iguales en X comparten casi la misma
    señal, así que su diferencia en y es casi puro ruido — la señal se
    cancela en la resta — y la resta de dos ruidos independientes tiene
    varianza 2σ²: de ahí el medio. (La varianza de y a secas no sirve
    de umbral: mezcla señal y ruido, y escalaría con la señal.)

    Vale en cualquier dimensión: X puede ser un vector (n,) o una
    matriz (n, d). En una dimensión el vecino más cercano es el
    contiguo al ordenar (estimador de Rice). Cuantas más dimensiones,
    más lejos queda el vecino y más señal se cuela en la diferencia:
    el estimador tiende a sobrestimar σ², y el umbral, a ser prudente.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) < 2:
        raise ValueError("X debe ser (n,) o (n, d) e y un vector (n,), con n >= 2.")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X e y deben ser finitos.")

    distancias = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
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

