"""El 3×3: tres estructuras × tres modelos, ordenados por error de señal.

Si el modelo con la estructura correcta gana (la diagonal), el ranking
diagnostica cómo se combinan las variables. Además, los ángulos de
Tangentes dicen cuál es la estructura aunque no se conozca.
"""
import numpy as np
import pandas as pd
from .estructuras import ESTRUCTURAS, muestra_2d
from .modelos_2d import (
        ajustar_diana, ajustar_pathfinder, ajustar_tangentes, ajustar_zigurat,
        )

MODELOS = {
    "Zigurat": ajustar_zigurat,
    "Diana": ajustar_diana,
    "Tangentes": ajustar_tangentes,
    }
EQUIVALENTE = {"aditiva": "Zigurat", "radial": "Diana", "direccional": "Tangentes"}
EXPLORADORES = {
    "Pathfinder": ajustar_pathfinder,
    }
#-- Los MODELOS son hipótesis sobre la estructura: el diagnóstico elige
#-- entre ellos. El Pathfinder explora el plano con la visión de
#-- Tangentes: aparece en la tabla de errores, fuera del concurso.


def una_repeticion(verdad, semilla, n=150, sigma=0.1, n_prueba=3000):
    X, y = muestra_2d(verdad, n, sigma, semilla)
    X_prueba, _ = muestra_2d(verdad, n_prueba, sigma, 10_000 + semilla)
    senal = verdad(X_prueba)
    #-- Error de señal ⟨|f − s|²⟩ en puntos nuevos del dominio, como en §3.2:
    #-- con la señal conocida, el ruido de prueba solo añadiría niebla.
    errores, angulos = {}, None
    for nombre, ajustar in {**MODELOS, **EXPLORADORES}.items():
        modelo = ajustar(X, y)
        errores[nombre] = np.mean((senal - modelo.predecir(X_prueba)) ** 2)
        if nombre == "Tangentes":
            angulos = modelo.angulos_grados()
    return errores, angulos


def ejecutar(repeticiones=10, **opciones):
    """Devuelve (error medio, victorias por modelo, ángulos de Tangentes)."""
    medias, victorias, angulos = {}, {}, {}
    for estructura, verdad in ESTRUCTURAS.items():
        resultados = [una_repeticion(verdad, s, **opciones) for s in range(repeticiones)]
        tabla = pd.DataFrame([errores for errores, _ in resultados])
        medias[estructura] = tabla.mean()
        victorias[estructura] = tabla[list(MODELOS)].idxmin(axis=1).value_counts().reindex(MODELOS, fill_value=0)
            #-- Solo las hipótesis concursan: el explorador queda fuera del diagnóstico.
        angulos[estructura] = [a for _, a in resultados]
    return pd.DataFrame(medias).T, pd.DataFrame(victorias).T, angulos
