"""Estimación del ruido y calibración del umbral de parada.

¿Cuándo deja de merecer la pena añadir una rampa? Cuando su mejora es
la que se conseguiría ajustando ruido puro. Para saber cuánto es eso
no se supone ninguna distribución: se mide en los propios datos.

1. MUESTRAS DEL RUIDO. Con el vecino más cercano en X, la diferencia
   d = (y_i − y_vecino)/√2 es casi puro ruido: dos observaciones casi
   iguales en X comparten casi la misma señal, que se cancela en la
   resta (por eso la varianza de y a secas no sirve: mezcla señal y
   ruido). Las d_i llevan la distribución real del ruido — sus colas,
   su escala — sin postular que sea gaussiana.

2. CALIBRACIÓN POR REMUESTREO (bootstrap). Se fabrican vectores de
   ruido puro remuestreando las d_i con reemplazo, se ejecuta sobre
   cada uno UN PASO DE LA PROPIA BÚSQUEDA, y el umbral es el cuantil
   `confianza` (estándar: 95%) de lo que la búsqueda robó. Así el
   umbral incorpora, sin fórmulas, el tamaño efectivo de la búsqueda
   (el máximo entre muchos candidatos correlacionados roba más que un
   candidato suelto) y vale en cualquier dimensión. Se calcula UNA
   SOLA VEZ, antes de la búsqueda.

Para ruido gaussiano existe el atajo clásico χ²₁(confianza)·σ̂²
(`umbral_de_mejora`), que ignora el tamaño de la búsqueda; la
calibración lo sustituye como criterio base.
"""
import numpy as np
from scipy.stats import chi2

REMUESTRAS = 50
#-- Repeticiones de la calibración: su cuantil 95 es grueso pero
#-- estable, y el coste total es ~50 pasos de búsqueda, una vez.


def diferencias_de_ruido(X, y):
    """Muestras del ruido: (y_i − y_vecino)/√2, con el vecino más cercano en X.

    El ruido se mide SIEMPRE en y; X solo decide quién es vecino de
    quién, y puede ser un vector (n,) o una matriz (n, d). En una
    dimensión el vecino más cercano es el contiguo al ordenar.
    Cuantas más dimensiones, más lejos queda el vecino y más señal se
    cuela en la diferencia: las muestras tienden a sobrestimar el
    ruido, y el umbral, a ser prudente.
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
    muestras = (y - y[vecino]) / np.sqrt(2.0)
    return muestras - muestras.mean()


def varianza_del_ruido(X, y):
    """σ̂²: la varianza de las muestras de ruido."""
    return float(np.mean(diferencias_de_ruido(X, y) ** 2))


def calibrar_umbral(X, y, robo_de_un_paso, confianza=0.95, semilla=0):
    """El umbral de mejora, calibrado con la propia búsqueda sobre ruido puro.

    `robo_de_un_paso(X, ruido)` debe ejecutar un paso de la búsqueda
    sobre un vector de ruido puro y devolver cuánto error elimina.
    Se llama REMUESTRAS veces con remuestreos de las diferencias de
    ruido de los datos, y el umbral es el cuantil `confianza` de los
    robos: una mejora real debe superar lo que la búsqueda le saca al
    ruido. La semilla fija hace la calibración determinista.
    """
    confianza = float(confianza)
    if not np.isfinite(confianza) or not 0.0 < confianza < 1.0:
        raise ValueError("confianza debe estar entre 0 y 1 (abiertos).")

    muestras = diferencias_de_ruido(X, y)
    generador = np.random.default_rng(semilla)
    robos = [
        float(robo_de_un_paso(X, generador.choice(muestras, size=len(muestras))))
        for _ in range(REMUESTRAS)
    ]
    return float(np.quantile(robos, confianza))


def umbral_de_mejora(varianza, confianza=0.95):
    """Atajo gaussiano: χ²₁(confianza)·σ̂², la mejora que roba UN grado
    de libertad ajustado a ruido normal. Supone ruido gaussiano e
    ignora que la búsqueda elige el mejor entre muchos candidatos;
    `calibrar_umbral` no supone ni ignora ninguna de las dos cosas.
    """
    confianza = float(confianza)
    if not np.isfinite(confianza) or not 0.0 < confianza < 1.0:
        raise ValueError("confianza debe estar entre 0 y 1 (abiertos).")
    varianza = float(varianza)
    if not np.isfinite(varianza) or varianza < 0:
        raise ValueError("La varianza debe ser finita y no negativa.")
    return float(chi2.ppf(confianza, df=1) * varianza)
