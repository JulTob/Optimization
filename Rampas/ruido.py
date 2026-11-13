"""Ruido, blancura y umbrales: los criterios de parada.

CRITERIO BASE — BLANCURA DEL RESIDUO (señales y sistemas). Un modelo
ha extraído toda la estructura cuando su residuo es ruido blanco: su
autocorrelación es una delta. Si queda estructura monótona por
capturar, los residuos vecinos se parecen y la autocorrelación a
retardo 1 sale positiva; si solo queda ruido, cae dentro de la banda
±banda/√n, que se calcula del tamaño de la muestra, en forma cerrada,
sin suponer la distribución del ruido (banda = 2 equivale a dos
desviaciones típicas de ρ₁ bajo blancura, ≈ 95%). La búsqueda sigue
mientras el residuo tenga color y se detiene cuando es blanco.

ALTERNATIVAS, conservadas para compararlas como resultado (§4):
- umbral_de_mejora: atajo gaussiano χ²₁(confianza)·σ̂², que supone
  ruido normal e ignora que la búsqueda elige el mejor candidato.
- calibrar_umbral: remuestrea el ruido de los propios datos y ejecuta
  la búsqueda real sobre él; exacto y libre de distribución, pero
  simulado en vez de cerrado.

El ruido se mide SIEMPRE en y; X solo decide quién es vecino de quién:
dos observaciones casi iguales en X comparten casi la misma señal, que
se cancela en la resta (la varianza de y a secas mezcla señal y ruido).
"""
import numpy as np
from scipy.stats import chi2

def autocorrelacion_primera(residuos):
    """ρ₁ = Σ eᵢ·eᵢ₊₁ / Σ eᵢ², con los residuos en su orden espacial."""
    residuos = np.asarray(residuos, dtype=float)
    if residuos.ndim != 1 or residuos.size < 2:
        raise ValueError("Hacen falta al menos 2 residuos en un vector.")
    if not np.all(np.isfinite(residuos)):
        raise ValueError("Los residuos deben ser finitos.")
    residuos = residuos - residuos.mean()
    energia = float(residuos @ residuos)
    if energia <= 0.0:
        return 0.0
        #-- Residuo idénticamente nulo: tan blanco como se puede ser.
    return float(residuos[:-1] @ residuos[1:]) / energia


def es_blanco(residuos, banda=2.0):
    """¿Es el residuo ruido blanco? ρ₁ ≤ banda/√n.

    La prueba es de un solo lado: la estructura monótona que una rampa
    puede capturar deja ρ₁ POSITIVA (residuos vecinos parecidos); una
    ρ₁ muy negativa sería sobreajuste, que tampoco pide más rampas.
    Con menos de 4 residuos no hay color que medir: se declara blanco.
    """
    banda = float(banda)
    if not np.isfinite(banda):
        raise ValueError("banda debe ser finita.")
    residuos = np.asarray(residuos, dtype=float)
    if residuos.size < 4:
        return True
    return autocorrelacion_primera(residuos) <= banda / np.sqrt(residuos.size)


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
