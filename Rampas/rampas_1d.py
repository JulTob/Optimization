"""Ajustes con suma de rampas en una variable: f(x) = b + Σ w_j R_j(x), w_j ≥ 0."""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import lsq_linear
from .senales import matriz_de_rampas
from .ruido import umbral_de_mejora, varianza_del_ruido


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


def rampas_por_division(x, y, profundidad=None, confianza=0.95):
    """Divide el dominio recursivamente hasta que partir no supere al ruido.

    Se empieza con una sola rampa, de min(x) a max(x). En cada intervalo
    se busca el nudo interior que más reduce el error, y el corte se
    acepta si su mejora supera el UMBRAL DE RUIDO: el error que robaría
    un grado de libertad ajustado a ruido puro, al nivel de confianza
    dado (χ²₁(confianza)·σ̂²; ver Rampas.ruido). σ̂² se estima una sola
    vez, antes de la búsqueda. Aceptado el corte, la búsqueda se repite
    dentro de [izquierda, corte] y de [corte, derecha]: el número de
    rampas EMERGE de la búsqueda.

    `profundidad` limita los niveles de recursión: el nivel k permite
    hasta 2^k − 1 cortes. Con None no hay tope, y la recursión acaba
    igualmente: los candidatos son los datos, finitos, y las mejoras se
    agotan porque el error nunca baja del de la isotónica (§2.3).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if profundidad is not None and (
            not isinstance(profundidad, (int, np.integer)) or profundidad < 1
            ):
        raise ValueError("profundidad debe ser un entero mayor o igual que 1, o None.")

    posiciones = np.unique(x)
    if posiciones.size < 2:
        raise ValueError("Hacen falta al menos dos valores de x distintos.")

    umbral = umbral_de_mejora(varianza_del_ruido(x, y), confianza)
    cortes = [float(posiciones[0]), float(posiciones[-1])]
    _dividir(x, y, cortes, cortes[0], cortes[1], profundidad, umbral)
    return _ajustar_entre_cortes(x, y, sorted(cortes))


def _dividir(x, y, cortes, izquierda, derecha, profundidad, umbral):
    """Busca un corte dentro de (izquierda, derecha); si supera al ruido, recurre."""
    if profundidad is not None and profundidad < 1:
        return
    candidatos = [float(c) for c in np.unique(x) if izquierda < c < derecha]
    if not candidatos:
        return

    error_actual = _error(x, y, _ajustar_entre_cortes(x, y, sorted(cortes)))
    if error_actual <= 0:
        return
        #-- Ajuste ya exacto: no hay nada que mejorar.

    errores = [
        _error(x, y, _ajustar_entre_cortes(x, y, sorted(cortes + [c])))
        for c in candidatos
    ]
    mejor = int(np.argmin(errores))
    if error_actual - errores[mejor] <= umbral:
        return
        #-- La mejora no supera lo que robaría el ruido: punto estable.

    corte = candidatos[mejor]
    cortes.append(corte)
    siguiente = None if profundidad is None else profundidad - 1
    _dividir(x, y, cortes, izquierda, corte, siguiente, umbral)
    _dividir(x, y, cortes, corte, derecha, siguiente, umbral)


def _ajustar_entre_cortes(x, y, cortes):
    cortes = np.asarray(cortes)
    return ajustar_pesos(x, y, cortes[:-1], np.diff(cortes))


def _error(x, y, ajuste):
    return float(np.sum((y - ajuste(x)) ** 2))
