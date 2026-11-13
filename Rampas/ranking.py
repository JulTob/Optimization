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

    senal, test = _errores_por_repeticion(
            s, bajo, alto, sigma, n, n_ajustes, n_pruebas, n_malla, semilla, modelos,
            )
    filas = []
    for nombre in modelos:
        filas.append({
            "Modelo": nombre,
            "Test (media)": test[nombre].mean(),
            "Test (varianza)": test[nombre].var(ddof=1),
            "Señal (media)": senal[nombre].mean(),
            "Señal (varianza)": senal[nombre].var(ddof=1),
            "Ajustes fallidos": int(senal[nombre].isna().sum()),
        })
    tabla = pd.DataFrame(filas).set_index("Modelo")
    return tabla.sort_values("Señal (media)", na_position="last")


def tabla_de_victorias(
        s, bajo, alto, sigma,
        n=25,
        n_ajustes=10,
        n_malla=500,
        semilla=0,
        modelos=None,
        ):
    """Tabla de victorias: puesto de cada modelo, repetición a repetición.

    En cada repetición los modelos se ordenan por su error de señal
    ⟨|f(x) − s(x)|²⟩ (el ajuste a la señal generadora, sin ruido de
    prueba). La tabla da las VICTORIAS (veces en el puesto 1), el
    puesto medio, el mejor y el peor, y se ordena por puesto medio.
    Un ajuste fallido no puntúa en su repetición.
    """
    if modelos is None:
        modelos = {**catalogo_de_rampas(), **catalogo_de_referencias()}
    senal, _ = _errores_por_repeticion(
            s, bajo, alto, sigma, n, n_ajustes, 2, n_malla, semilla, modelos,
            )
    puestos = senal.rank(axis=1, method="min")
    #-- Puesto 1 = menor error de señal en esa repetición; los empates
    #-- comparten el mejor puesto.
    filas = []
    for nombre in modelos:
        filas.append({
            "Modelo": nombre,
            "Victorias": int((puestos[nombre] == 1).sum()),
            "Puesto medio": puestos[nombre].mean(),
            "Mejor": puestos[nombre].min(),
            "Peor": puestos[nombre].max(),
        })
    tabla = pd.DataFrame(filas).set_index("Modelo")
    return tabla.sort_values("Puesto medio", na_position="last")


def _errores_por_repeticion(
        s, bajo, alto, sigma, n, n_ajustes, n_pruebas, n_malla, semilla, modelos,
        ):
    """Errores por repetición, con las MISMAS muestras para todos los modelos.

    Devuelve dos tablas con una columna por modelo: la de señal (una
    fila por ajuste: ⟨|f − s|²⟩ sobre la malla) y la de test (una fila
    por muestra de prueba: ⟨|y − f(x)|²⟩). Un ajuste fallido deja NaN
    en sus filas.
    """
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
    senal_en_malla = np.asarray(s(malla), dtype=float)

    senal = pd.DataFrame(index=range(n_ajustes), columns=list(modelos), dtype=float)
    test = pd.DataFrame(
            index=range(n_ajustes * n_pruebas), columns=list(modelos), dtype=float,
            )
    for nombre, ajustar in modelos.items():
        for repeticion, ((x, y), muestras_de_prueba) in enumerate(zip(entrenamientos, pruebas)):
            try:
                f = ajustar(x, y)
            except (ValueError, RuntimeError):
                continue
                #-- El fallo queda como NaN: no puntúa ni promedia.
            senal.loc[repeticion, nombre] = float(np.mean((f(malla) - senal_en_malla) ** 2))
            for indice, (x_prueba, y_prueba) in enumerate(muestras_de_prueba):
                test.loc[repeticion * n_pruebas + indice, nombre] = float(
                        np.mean((y_prueba - f(x_prueba)) ** 2)
                        )
    return senal, test
