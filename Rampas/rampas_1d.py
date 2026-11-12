"""Ajustes con suma de rampas en una variable: f(x) = b + Σ w_j R_j(x), w_j ≥ 0."""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import lsq_linear
from .senales import matriz_de_rampas


@dataclass(frozen=True)
class SumaDeRampas:
    base: float
    inicios: np.ndarray
    anchos: np.ndarray
    pesos: np.ndarray

    def __call__(self, x):
        return self.base + matriz_de_rampas(x, self.inicios, self.anchos) @ self.pesos


def ajustar_pesos(x, y, inicios, anchos, penalizacion=0.0):
    """Mínimos cuadrados con base libre y pesos ≥ 0 (penalización opcional λ·Σw²)."""
    inicios = np.asarray(inicios, dtype=float)
    anchos = np.broadcast_to(np.asarray(anchos, dtype=float), inicios.shape)
    if np.any(anchos <= 0):
        raise ValueError("Las anchuras deben ser positivas.")
    rampas = matriz_de_rampas(x, inicios, anchos)
    diseno = np.column_stack([np.ones(len(x)), rampas])
    objetivo = np.asarray(y, dtype=float)
    if penalizacion > 0:
        # Un ridge es el mismo problema con filas extra: √λ·w ≈ 0.
        filas_extra = np.column_stack([
                np.zeros(len(inicios)),
                np.sqrt(penalizacion) * np.eye(len(inicios)),
                ])
        diseno = np.vstack([diseno, filas_extra])
        objetivo = np.r_[objetivo, np.zeros(len(inicios))]
    cotas_inferiores = np.r_[-np.inf, np.zeros(len(inicios))]
    solucion = lsq_linear(
            diseno,
            objetivo,
            bounds=(cotas_inferiores, np.inf),
            method="bvls",
            tol=1e-12,
            )
    if not solucion.success:
        raise RuntimeError(f"El ajuste no convergió: {solucion.message}")
    return SumaDeRampas(
            float(solucion.x[0]),
            inicios,
            np.array(anchos),
            solucion.x[1:],
            )


def rampas_con_nudo_en_cada_dato(x, y):
    """Isotónica lineal: una rampa entre cada par de valores distintos de x."""
    nudos = np.unique(x)
    return ajustar_pesos(x, y, nudos[:-1], np.diff(nudos))


def rampas_comprimidas(x, y, n_rampas=6):
    """K rampas entre K+1 nudos equiespaciados de min(x) a max(x)."""
    nudos = np.linspace(np.min(x), np.max(x), n_rampas + 1)
    return ajustar_pesos(x, y, nudos[:-1], np.diff(nudos))


def rampas_regularizadas(x, y, n_anchuras=4, n_posiciones=12, penalizacion=1e-3):
    """Diccionario multiescala: anchuras (1, 1/2, 1/4, …)·rango; pesos ≥ 0 y λΣw²."""
    minimo, rango = np.min(x), np.ptp(x)
    inicios, anchos = [], []
    for k in range(n_anchuras):
        ancho = rango / 2**k
        posiciones = np.linspace(minimo, minimo + rango - ancho, n_posiciones)
        inicios.extend(posiciones)
        anchos.extend([ancho] * n_posiciones)
    return ajustar_pesos(x, y, inicios, anchos, penalizacion)
