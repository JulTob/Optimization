"""Camino por división: la división de rampas de §2.4, aplicada a un camino.

En una variable, la división parte un tramo [a, b] en dos tramos que
suman lo mismo. Aquí se parte un SEGMENTO del camino, de A a B, por un
punto M: los dos vectores nuevos, M − A y B − M, suman el original
(siguen siendo la diagonal de la caja entre A y B). Cada segmento lleva
una rampa cuyas curvas de nivel son hiperplanos perpendiculares a él.

Los dos vectores son válidos (no negativos en cada coordenada, la
condición de monotonía) si y solo si M está dentro de la caja entre A
y B. Buscar M en esa caja es buscar la variación del camino: a lo largo
del segmento y en su hiperplano tangente. Nada de esto depende de la
dimensión; aquí la caja se recorre con una malla, que basta en el plano
(en dimensión alta la malla debería sustituirse por un gradiente).

Se empieza con el segmento diagonal del dominio normalizado, de 0 a 1
en cada coordenada, y cada segmento se parte, recursivamente, mientras
la mejora relativa del error supere el umbral ε.
"""
import itertools

import numpy as np
from scipy.optimize import lsq_linear

from .aditivo import Componente, Proyeccion
from .dominio import Escala
from .isotonica import Nudos
from .modelos_2d import Modelo2D


def ajustar_camino_por_division(X, z, profundidad=None, umbral=None, divisiones=10):
    """Ajusta un camino de rampas partiendo segmentos desde la diagonal.

    `umbral` es la mejora relativa mínima de un corte. Con None se usa el
    estándar ε = 2·Δp/n, con Δp = d + 1 (las d coordenadas de M y un
    peso). `profundidad` limita los niveles de recursión. `divisiones`
    es la finura de la malla con que se busca M dentro de cada caja.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("z debe ser un vector finito no vacío.")
    if profundidad is not None and (
            not isinstance(profundidad, (int, np.integer)) or profundidad < 1
            ):
        raise ValueError("profundidad debe ser un entero mayor o igual que 1, o None.")
    if not isinstance(divisiones, (int, np.integer)) or divisiones < 2:
        raise ValueError("divisiones debe ser un entero mayor o igual que 2.")

    escala = Escala.de_los_datos(X)
    Z = escala(X)
    if len(Z) != len(z):
        raise ValueError("X y z deben tener el mismo número de datos.")
    dimension = Z.shape[1]
    if umbral is None:
        umbral = 2.0 * (dimension + 1) / len(z)
        #-- AIC: 2·Δp/n con Δp = d + 1 (el punto M y un peso).
    umbral = float(umbral)
    if not np.isfinite(umbral) or umbral < 0:
        raise ValueError("umbral debe ser finito y no negativo.")

    camino = [np.zeros(dimension), np.ones(dimension)]
    #-- El segmento inicial: la diagonal del dominio normalizado.
    rejilla = _rejilla_de_la_caja(dimension, divisiones)
    _dividir_segmento(Z, z, camino, 0, profundidad, umbral, rejilla)

    base, pesos = _pesos(Z, z, camino)
    return Modelo2D("Camino por división", escala, _componentes(camino, base, pesos))


def puntos_del_camino(modelo):
    """Los vértices P₀, P₁, … del camino, en el dominio normalizado."""
    puntos = []
    for componente in modelo.componentes:
        if getattr(componente.proyeccion, "vector", None) is None:
            continue
        inicio, fin = componente.proyeccion.extremos
        if not puntos:
            puntos.append(inicio)
        puntos.append(fin)
    return np.array(puntos)


def _dividir_segmento(Z, z, camino, indice, profundidad, umbral, rejilla):
    """Parte el segmento camino[indice] → camino[indice + 1] si compensa."""
    if profundidad is not None and profundidad < 1:
        return
    inicio, fin = camino[indice], camino[indice + 1]
    if np.linalg.norm(fin - inicio) < 1e-9:
        return

    error_actual = _error(Z, z, camino)
    if error_actual <= 0:
        return
        #-- Ajuste ya exacto: no hay nada que mejorar.

    mejor_punto, mejor_error = None, np.inf
    for fraccion in rejilla:
        punto = inicio + fraccion * (fin - inicio)
        #-- M dentro de la caja entre A y B: los dos vectores son válidos.
        if (np.linalg.norm(punto - inicio) < 1e-9
                or np.linalg.norm(fin - punto) < 1e-9):
            continue
        candidato = camino[: indice + 1] + [punto] + camino[indice + 1:]
        error = _error(Z, z, candidato)
        if error < mejor_error:
            mejor_punto, mejor_error = punto, error
    if mejor_punto is None:
        return
    if (error_actual - mejor_error) / error_actual <= umbral:
        return
        #-- La mejora no paga su coste de descripción: punto estable.

    camino.insert(indice + 1, mejor_punto)
    siguiente = None if profundidad is None else profundidad - 1
    _dividir_segmento(Z, z, camino, indice + 1, siguiente, umbral, rejilla)
    _dividir_segmento(Z, z, camino, indice, siguiente, umbral, rejilla)
    #-- Primero el segundo tramo: así el índice del primero no se mueve.


def _rejilla_de_la_caja(dimension, divisiones):
    """Las fracciones τ ∈ [0, 1]^d con que se recorre la caja, M = A + τ·(B − A)."""
    pasos = np.linspace(0.0, 1.0, divisiones + 1)
    return [np.array(t) for t in itertools.product(pasos, repeat=dimension)]


def _rampas(Z, camino):
    """Una columna por segmento: la rampa que sube a lo largo de él."""
    columnas = []
    for inicio, fin in zip(camino[:-1], camino[1:]):
        vector = fin - inicio
        largo = float(np.linalg.norm(vector))
        unitario = vector / largo
        columnas.append(np.clip((Z - inicio) @ unitario / largo, 0.0, 1.0))
    return np.column_stack(columnas)


def _pesos(Z, z, camino):
    """Base libre y un peso ≥ 0 por segmento: mínimos cuadrados no negativos."""
    diseno = np.column_stack([np.ones(len(Z)), _rampas(Z, camino)])
    tramos = len(camino) - 1
    resultado = lsq_linear(
            diseno, z,
            bounds=(np.r_[-np.inf, np.zeros(tramos)], np.inf),
            method="bvls", tol=1e-12,
            )
    if not resultado.success:
        raise RuntimeError(f"El ajuste no ha convergido: {resultado.message}")
    return float(resultado.x[0]), resultado.x[1:]


def _error(Z, z, camino):
    base, pesos = _pesos(Z, z, camino)
    return float(np.sum((z - base - _rampas(Z, camino) @ pesos) ** 2))


class _ProyeccionDeSegmento(Proyeccion):
    """Proyección sobre la dirección de un segmento, con sus extremos."""

    def __init__(self, inicio, fin):
        vector = fin - inicio
        unitario = vector / np.linalg.norm(vector)
        angulo = (
                float(np.arctan2(unitario[1], unitario[0]))
                if len(unitario) == 2 else None
                )
        object.__setattr__(self, "nombre", "segmento")
        object.__setattr__(self, "funcion", lambda X: X @ unitario)
        object.__setattr__(self, "angulo", angulo)
        object.__setattr__(self, "vector", unitario)
        object.__setattr__(self, "extremos", (inicio, fin))


def _componentes(camino, base, pesos):
    """Cada segmento es una Componente de dos nudos; la base va en el primero."""
    componentes = []
    for k, ((inicio, fin), peso) in enumerate(zip(zip(camino[:-1], camino[1:]), pesos)):
        proyeccion = _ProyeccionDeSegmento(inicio, fin)
        a = float(proyeccion.vector @ inicio)
        b = float(proyeccion.vector @ fin)
        nivel = base if k == 0 else 0.0
        componentes.append(Componente(
                proyeccion, Nudos(np.array([a, b]), np.array([nivel, nivel + peso])),
                ))
    return componentes
