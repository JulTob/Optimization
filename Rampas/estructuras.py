"""Datos con estructura CONOCIDA en el cuadrante positivo [0,1]².

Son la verdad de partida del experimento: sabemos qué modelo debería ganar.
"""
import numpy as np
from scipy.special import expit

ANGULO_VERDADERO = np.radians(30)


def aditiva(X):
    return 0.5 * np.sqrt(X[:, 0]) + 0.5 * X[:, 1] ** 2


def radial(X):
    r = np.linalg.norm(X, axis=1) / np.sqrt(2)
    return np.sqrt(r)


def direccional(X):
    n = np.array([np.cos(ANGULO_VERDADERO), np.sin(ANGULO_VERDADERO)])
    return expit(8 * (X @ n - 0.5))


ESTRUCTURAS = {
    "aditiva": aditiva,
    "radial": radial,
    "direccional": direccional,
    }


def muestra_2d(verdad, n, sigma, semilla):
    generador = np.random.default_rng(semilla)
    X = generador.uniform(0.0, 1.0, size=(n, 2))
    y = verdad(X) + generador.normal(0.0, sigma, n)
    return X, y
