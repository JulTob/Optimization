"""Dibujo con matplotlib: una función por figura."""
import numpy as np
import matplotlib.pyplot as plt


def dibujar_1d(x, y, curvas, titulo=""):
    """Datos y varias curvas {nombre: función}."""
    malla = np.linspace(np.min(x), np.max(x), 400)
    figura, eje = plt.subplots(figsize=(7, 4))
    eje.scatter(x, y, color="black", s=18, label="datos")
    for nombre, curva in curvas.items():
        eje.plot(malla, curva(malla), label=nombre)
    eje.set(title=titulo, xlabel="x", ylabel="y")
    eje.legend()
    return figura


def dibujar_mapas_2d(X, y, modelos, verdad=None):
    """Un mapa de contornos por modelo (y por la verdad, si se da)."""
    paneles = ({"Verdad": verdad} if verdad else {}) | {
            m.nombre: m.predecir for m in modelos
            }
    lado = np.linspace(0, 1, 80)
    A, B = np.meshgrid(lado, lado)
    puntos = np.column_stack([A.ravel(), B.ravel()])
    figura, ejes = plt.subplots(1, len(paneles), figsize=(4 * len(paneles), 3.8), sharey=True)
    for eje, (nombre, funcion) in zip(np.atleast_1d(ejes), paneles.items()):
        relleno = eje.contourf(A, B, funcion(puntos).reshape(A.shape), levels=12)
        eje.scatter(X[:, 0], X[:, 1], c="white", s=5, alpha=0.6)
        eje.set(title=nombre, xlabel="x1")
    figura.colorbar(relleno, ax=ejes)
    return figura
