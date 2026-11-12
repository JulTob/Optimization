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


def rampas_libres(x, y, n_rampas=6, vueltas=10):
    """K rampas de cualquier anchura, elegidas entre todas las que van de un nudo a otro.

    Elegir el mejor subconjunto es combinatorio: se añade una rampa cada vez
    (la que más reduce el error) y después se repasa cada una por turnos,
    cambiándola por otra si mejora. El error nunca sube, así que termina.
    """
    nudos = np.unique(x)
    i, j = np.triu_indices(len(nudos), k=1)
    inicios, anchos = nudos[i], nudos[j] - nudos[i]
    elegidas = []
    for _ in range(min(n_rampas, len(inicios))):
        elegidas.append(_mejor_candidata(x, y, elegidas, inicios, anchos))
    for _ in range(vueltas):
        hubo_cambio = False
        for k in range(len(elegidas)):
            otras = elegidas[:k] + elegidas[k + 1:]
            nueva = _mejor_candidata(x, y, otras, inicios, anchos)
            hubo_cambio |= nueva != elegidas[k]
            elegidas[k] = nueva
        if not hubo_cambio:
            break
    return ajustar_pesos(x, y, inicios[elegidas], anchos[elegidas])


def _mejor_candidata(x, y, elegidas, inicios, anchos):
    """Índice de la rampa que, añadida a `elegidas`, deja menos error."""
    mejor, menor_error = None, np.inf
    for c in range(len(inicios)):
        if c in elegidas:
            continue
        indices = elegidas + [c]
        ajuste = ajustar_pesos(x, y, inicios[indices], anchos[indices])
        error = np.sum((y - ajuste(x)) ** 2)
        if error < menor_error - 1e-12:
            mejor, menor_error = c, error
    return mejor


def rampas_regularizadas(x, y, n_anchuras=4, penalizacion=1e-3):
    """Catálogo multiescala fijo; pesos ≥ 0 y penalización λΣw².

    Anchuras: el rango, la mitad, un cuarto, … Cada anchura se coloca
    desplazándola media anchura cada vez: 1, 3, 7, 15, … posiciones distintas.
    """
    minimo, rango = np.min(x), np.ptp(x)
    inicios, anchos = [], []
    for k in range(n_anchuras):
        ancho = rango / 2**k
        n_posiciones = 2 ** (k + 1) - 1
        inicios.extend(np.linspace(minimo, minimo + rango - ancho, n_posiciones))
        anchos.extend([ancho] * n_posiciones)
    return ajustar_pesos(x, y, inicios, anchos, penalizacion)
