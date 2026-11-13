"""Ranking de modelos por error de predicción (§3.2).

El experimento repite, para cada modelo, el ciclo entrenar → probar:

1. Se generan muestras de entrenamiento independientes y cada modelo
   se ajusta a cada una.
2. Cada ajuste se evalúa con muestras de prueba nuevas, con sus
   propias x: sobre las x de entrenamiento la isotónica escalonada y
   la lineal coinciden (§2.3) y no podrían distinguirse.
3. Todos los modelos ven exactamente las mismas muestras, de
   entrenamiento y de prueba: comparación emparejada.

Para cada modelo se reportan media y varianza del error de test
⟨|y − f(x)|²⟩ y del error de señal ⟨|f(x) − s(x)|²⟩, y la tabla se
ordena por error de señal medio.
"""

import numpy as np
import pandas as pd

from .isotonica import ajustar_isotonica, evaluar_escalones, evaluar_nudos
from .rampas_1d import (
    rampas_comprimidas,
    rampas_libres,
    rampas_por_division,
    rampas_regularizadas,
)
from .referencias import catalogo_de_referencias


def catalogo_de_rampas():
    """Las seis familias de §2.4, como funciones de ajuste (x, y) → predictor."""

    def escalonada(x, y):
        nudos = ajustar_isotonica(x, y)
        return lambda puntos: evaluar_escalones(nudos, puntos)

    def lineal(x, y):
        nudos = ajustar_isotonica(x, y)
        return lambda puntos: evaluar_nudos(nudos, puntos)
        #-- Igual que rampas con un nudo por dato (§2.3), por el camino rápido.

    return {
        "Isotónica escalonada": escalonada,
        "Isotónica lineal": lineal,
        "Rampas comprimidas (K = 6)": rampas_comprimidas,
        "Rampas libres (K = 6)": rampas_libres,
        "Rampas regularizadas": rampas_regularizadas,
        "Rampas por división": rampas_por_division,
    }


def muestra_1d(s, bajo, alto, sigma, n, generador):
    """Una muestra (x, y): x ~ U(bajo, alto) y y = s(x) + G, G ~ N(0, σ²)."""
    x = np.sort(generador.uniform(bajo, alto, n))
    y = np.asarray(s(x), dtype=float) + generador.normal(0.0, sigma, n)
    return x, y


def ranking_de_modelos(
        s, bajo, alto, sigma,
        n=25,
        n_ajustes=10,
        n_pruebas=10,
        n_malla=500,
        semilla=0,
        modelos=None,
        ):
    """Tabla de errores por modelo, ordenada por error de señal medio.

    Por modelo: n_ajustes ajustes, uno por muestra de entrenamiento,
    y n_ajustes·n_pruebas tests (n_pruebas muestras de prueba por
    ajuste). El error de señal se mide sobre una malla de n_malla
    puntos del dominio. 'Ajustes fallidos' cuenta los entrenamientos
    en los que el ajuste no convergió; sus tests no se cuentan.
    """
    sigma = float(sigma)
    bajo = float(bajo)
    alto = float(alto)
    if not (np.isfinite(bajo) and np.isfinite(alto)) or bajo >= alto:
        raise ValueError("Hace falta un dominio finito con bajo < alto.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma debe ser finito y no negativo.")
    for nombre, valor in (
            ("n", n),
            ("n_ajustes", n_ajustes),
            ("n_pruebas", n_pruebas),
            ("n_malla", n_malla),
            ):
        if not isinstance(valor, (int, np.integer)) or valor < 2:
            raise ValueError(f"{nombre} debe ser un entero mayor o igual que 2.")
    if modelos is None:
        modelos = {**catalogo_de_rampas(), **catalogo_de_referencias()}
    if not modelos:
        raise ValueError("Hace falta al menos un modelo.")

    generador = np.random.default_rng(semilla)
    entrenamientos = [
        muestra_1d(s, bajo, alto, sigma, n, generador)
        for _ in range(n_ajustes)
    ]
    pruebas = [
        [muestra_1d(s, bajo, alto, sigma, n, generador) for _ in range(n_pruebas)]
        for _ in range(n_ajustes)
    ]
    #-- Las muestras se generan antes de recorrer los modelos:
    #-- todos comparan exactamente sobre las mismas.

    malla = np.linspace(bajo, alto, n_malla)
    senal = np.asarray(s(malla), dtype=float)

    filas = []
    for nombre, ajustar in modelos.items():
        errores_test = []
        errores_senal = []
        fallos = 0
        for (x, y), muestras_de_prueba in zip(entrenamientos, pruebas):
            try:
                f = ajustar(x, y)
            except (ValueError, RuntimeError):
                fallos += 1
                continue
            errores_senal.append(float(np.mean((f(malla) - senal) ** 2)))
            for x_prueba, y_prueba in muestras_de_prueba:
                errores_test.append(float(np.mean((y_prueba - f(x_prueba)) ** 2)))
        filas.append({
            "Modelo": nombre,
            "Test (media)": _media(errores_test),
            "Test (varianza)": _varianza(errores_test),
            "Señal (media)": _media(errores_senal),
            "Señal (varianza)": _varianza(errores_senal),
            "Ajustes fallidos": fallos,
        })

    tabla = pd.DataFrame(filas).set_index("Modelo")
    return tabla.sort_values("Señal (media)", na_position="last")


def _media(valores):
    if not valores:
        return float("nan")
    return float(np.mean(valores))


def _varianza(valores):
    if len(valores) < 2:
        return float("nan")
    return float(np.var(valores, ddof=1))
