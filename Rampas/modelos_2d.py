"""Tres modelos en dos variables, con la MISMA pieza (suma de rampas).

    Zigurat    entran las coordenadas x1 y x2.      f = g1(x1) + g2(x2)
    Diana      entra el radio.                      f = g(‖x‖)
    Tangentes  entran direcciones libres, una por rampa o grupo de rampas.

Todos son crecientes en el cuadrante positivo (pesos ≥ 0, direcciones ≥ 0).
Para Tangentes la monotonía está garantizada; la optimalidad, no
(el error no es convexo en los ángulos): por eso se barre el ángulo.
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from .dominio import Escala
from .isotonica import ajustar_isotonica, evaluar_nudos
from .aditivo import (
        Componente, ajustar_componente, direccion, eje, radio, retroajustar, sumar,
        )

CUARTO_DE_VUELTA = np.pi / 2


@dataclass(frozen=True)
class Modelo2D:
    nombre: str
    escala: Escala
    componentes: list

    def predecir(self, X):
        return sumar(self.componentes, self.escala(X))

    def angulos_grados(self):
        """Direcciones aprendidas (solo Tangentes): es el DIAGNÓSTICO."""
        return sorted(
                round(np.degrees(c.proyeccion.angulo))
                for c in self.componentes
                if c.proyeccion.angulo is not None
                )


# ---------------------------------------------------------------- Zigurat y Diana
def ajustar_zigurat(X, y):
    escala = Escala.de_los_datos(X)
    componentes = retroajustar([eje(0), eje(1)], escala(X), y)
    return Modelo2D("Zigurat", escala, componentes)


def ajustar_diana(X, y):
    escala = Escala.de_los_datos(X)
    componentes = retroajustar([radio()], escala(X), y)
    return Modelo2D("Diana", escala, componentes)


# ---------------------------------------------------------------- Tangentes
def error_de_la_direccion(angulo, X, residuo):
    """Suma de cuadrados tras una isotónica del residuo sobre X·n(θ)."""
    n = np.array([np.cos(angulo), np.sin(angulo)])
    nudos = ajustar_isotonica(X @ n, residuo)
    return np.sum((residuo - evaluar_nudos(nudos, X @ n)) ** 2)


def mejor_angulo(X, residuo, n_barrido=45):
    """Barre de 0° a 90° y afina alrededor del mejor (sin quedarse en la rejilla)."""
    angulos = np.linspace(0.0, CUARTO_DE_VUELTA, n_barrido)
    errores = [error_de_la_direccion(a, X, residuo) for a in angulos]
    k = int(np.argmin(errores))
    paso = angulos[1] - angulos[0]
    entorno = (max(0.0, angulos[k] - paso), min(CUARTO_DE_VUELTA, angulos[k] + paso))
    afinado = minimize_scalar(
            error_de_la_direccion,
            bounds=entorno,
            args=(X, residuo),
            method="bounded",
            )
    return afinado.x if afinado.fun < errores[k] else angulos[k]


def retroajustar_direcciones(angulos, X, y, vueltas=10):
    """Cada dirección se reajusta (ángulo incluido) al residuo de las demás."""
    componentes = [None] * len(angulos)
    angulos = list(angulos)
    for _ in range(vueltas):
        antes = sumar(componentes, X)
        for j in range(len(angulos)):
            otras = componentes[:j] + [None] + componentes[j + 1:]
            residuo = y - sumar(otras, X)
            angulos[j] = mejor_angulo(X, residuo)
            componentes[j] = ajustar_componente(direccion(angulos[j]), X, residuo)
        if np.max(np.abs(sumar(componentes, X) - antes)) < 1e-6:
            break
    return angulos, componentes


def ajustar_tangentes(X, y, max_direcciones=6, fraccion_validacion=0.25, semilla=0):
    """Añade direcciones mientras mejore el error de PREDICCIÓN (validación)."""
    escala = Escala.de_los_datos(X)
    Z = escala(X)
    # 1. Reparto entrenamiento / validación, solo para decidir cuántas direcciones.
    orden = np.random.default_rng(semilla).permutation(len(Z))
    n_val = int(len(Z) * fraccion_validacion)
    val, ent = orden[:n_val], orden[n_val:]
    # 2. Crecer una dirección cada vez.
    angulos, componentes = [], []
    mejor_error = np.mean((y[val] - y[ent].mean()) ** 2)
    for _ in range(max_direcciones):
        residuo = y[ent] - sumar(componentes, Z[ent])
        nuevo = mejor_angulo(Z[ent], residuo)
        propuestos, candidatas = retroajustar_direcciones(angulos + [nuevo], Z[ent], y[ent])
        error = np.mean((y[val] - sumar(candidatas, Z[val])) ** 2)
        if error >= mejor_error:
            break
        angulos, componentes, mejor_error = propuestos, candidatas, error
    # 3. Reajuste final con todos los datos y el número de direcciones elegido.
    if not angulos:
        angulos = [mejor_angulo(Z, y - y.mean())]
    angulos, componentes = retroajustar_direcciones(angulos, Z, y)
    return Modelo2D("Tangentes", escala, componentes)

