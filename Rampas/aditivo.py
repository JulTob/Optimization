"""Suma de componentes, cada una isotónica sobre una proyección de x.

Una componente es: u = proyección(x)  →  isotónica creciente de u  →  rampas.
Zigurat, Diana y Tangentes solo se distinguen en QUÉ proyección entra.
Se ajustan por turnos (backfitting): cada componente se reajusta a lo que
las demás no explican, hasta que la predicción se estabiliza.
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable
from .isotonica import ajustar_isotonica, evaluar_nudos


@dataclass(frozen=True)
class Proyeccion:
    nombre: str
    funcion: Callable
    angulo: float = None    #-- Solo si es una dirección (en radianes).

    def __call__(self, X):
        return self.funcion(X)


def eje(k):
    return Proyeccion(f"x{k + 1}", lambda X: X[:, k])


def radio():
    return Proyeccion("radio", lambda X: np.linalg.norm(X, axis=1))


def direccion(angulo):
    """n = (cos θ, sen θ) con θ en [0°, 90°]: componentes ≥ 0 (cono positivo)."""
    n = np.array([np.cos(angulo), np.sin(angulo)])
    return Proyeccion(f"{np.degrees(angulo):.0f}°", lambda X: X @ n, angulo)


@dataclass(frozen=True)
class Componente:
    proyeccion: Proyeccion
    nudos: object

    def __call__(self, X):
        return evaluar_nudos(self.nudos, self.proyeccion(X))

    def amplitud(self):
        """Cuánto sube esta componente de punta a punta: su peso en el modelo."""
        return float(self.nudos.niveles[-1] - self.nudos.niveles[0])


def ajustar_componente(proyeccion, X, residuo):
    return Componente(proyeccion, ajustar_isotonica(proyeccion(X), residuo))


def sumar(componentes, X):
    """Suma de las componentes ya ajustadas (las None aún no existen)."""
    total = np.zeros(len(X))
    for componente in componentes:
        if componente is not None:
            total += componente(X)
    return total


def retroajustar(proyecciones, X, y, vueltas=100, tolerancia=1e-9):
    """Ajusta por turnos una componente por proyección."""
    componentes = [None] * len(proyecciones)
    for _ in range(vueltas):
        antes = sumar(componentes, X)
        for j, proyeccion in enumerate(proyecciones):
            otras = componentes[:j] + [None] + componentes[j + 1:]
            residuo = y - sumar(otras, X)
            componentes[j] = ajustar_componente(proyeccion, X, residuo)
        despues = sumar(componentes, X)
        if np.max(np.abs(despues - antes)) < tolerancia:
            break
    return componentes
