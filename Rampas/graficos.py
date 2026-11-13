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


def dibujar_diagnostico(diagnostico, X, y, verdad=None, resolucion=80):
    """El original (verdad o datos) y los tres modelos, en las unidades de los datos.

    Cada panel lleva su error de predicción; el ganador, una ★.
    En Tangentes se dibujan las direcciones relevantes, con su aporte.
    """
    escala = diagnostico.modelos["Zigurat"].escala
    lado = np.linspace(0, 1, resolucion)
    A, B = np.meshgrid(lado, lado)
    unitarios = np.column_stack([A.ravel(), B.ravel()])
    puntos = escala.minimo + escala.rango * unitarios
    U = puntos[:, 0].reshape(A.shape)
    V = puntos[:, 1].reshape(A.shape)

    superficies = {"Verdad" if verdad else "Datos": verdad(puntos) if verdad else None}
    for nombre in diagnostico.orden():
        superficies[nombre] = diagnostico.modelos[nombre].predecir(puntos)
    niveles = np.linspace(min(np.min(y), *(np.min(z) for z in superficies.values() if z is not None)),
                          max(np.max(y), *(np.max(z) for z in superficies.values() if z is not None)), 13)

    figura, ejes = plt.subplots(1, 4, figsize=(17, 4), sharey=True, constrained_layout=True)
    for eje, (nombre, valores) in zip(ejes, superficies.items()):
        if valores is None:
            relleno = eje.scatter(X[:, 0], X[:, 1], c=y, vmin=niveles[0], vmax=niveles[-1], s=14)
            eje.set_title("Datos originales")
        else:
            relleno = eje.contourf(U, V, valores.reshape(A.shape), levels=niveles)
            eje.scatter(X[:, 0], X[:, 1], c="white", s=4, alpha=0.5)
            eje.set_title(_titulo(nombre, diagnostico))
        eje.set_xlabel("x1")
    ejes[0].set_ylabel("x2")
    _dibujar_direcciones(ejes[list(superficies).index("Tangentes")], diagnostico, escala)
    figura.colorbar(relleno, ax=ejes, shrink=0.85)
    figura.suptitle(diagnostico.lectura(), fontsize=9, wrap=True)
    return figura


def _escribir_errores(eje, x, y, curva, malla, verdad):
    """ECM de entrenamiento y, si se conoce, ECM frente a la verdad, en la esquina inferior derecha."""
    lineas = [f"ECM entrenamiento: {np.mean((y - curva(x)) ** 2):.4f}"]
    if verdad is not None:
        lineas.append(f"ECM frente a la verdad: {np.mean((verdad(malla) - curva(malla)) ** 2):.4f}")
    eje.text(
            0.97, 0.04, "\n".join(lineas),
            transform=eje.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
            )


def _titulo(nombre, diagnostico):
    if nombre not in diagnostico.errores:
        return "Verdad (función original)"
    marca = " ★" if nombre == diagnostico.ganador() else ""
    return f"{nombre}{marca} · ECM {diagnostico.errores[nombre]:.4f}"


def _dibujar_direcciones(eje, diagnostico, escala):
    """Flechas desde el mínimo de los datos, con longitud proporcional al aporte."""
    for grados, aporte in diagnostico.direcciones:
        n = np.array([np.cos(np.radians(grados)), np.sin(np.radians(grados))])
        fin = escala.minimo + escala.rango * (0.9 * aporte * n)
        eje.annotate(
                "",
                xy=fin, xytext=escala.minimo,
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
                )
        eje.text(*fin, f" {grados:.0f}° ({aporte:.0%})", color="red", fontsize=8)


def dibujar_familias(x, y, verdad=None, titulo=""):
    """Las familias de §2.4 con los mismos datos y la misma escala."""
    from .isotonica import ajustar_isotonica, evaluar_escalones, evaluar_nudos
    from .rampas_1d import rampas_comprimidas, rampas_libres, rampas_por_division, rampas_regularizadas

    nudos = ajustar_isotonica(x, y)
    familias = {
        "Isotónica escalonada": lambda t: evaluar_escalones(nudos, t),
        "Isotónica lineal": lambda t: evaluar_nudos(nudos, t),
        "Rampas comprimidas (K = 6)": rampas_comprimidas(x, y),
        "Rampas libres (K = 6)": rampas_libres(x, y),
        "Rampas regularizadas": rampas_regularizadas(x, y),
        }
    por_division = rampas_por_division(x, y)
    familias[f"Rampas por división (K = {len(por_division.pesos)}, validación)"] = por_division
    malla = np.linspace(np.min(x), np.max(x), 1000)
    figura, ejes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True)
    for eje, (nombre, curva) in zip(ejes.ravel(), familias.items()):
        eje.scatter(x, y, color="black", s=14, zorder=3)
        if verdad is not None:
            eje.plot(malla, verdad(malla), "--", color="gray", label="verdad")
        eje.plot(malla, curva(malla), color="tab:red", lw=2, label="ajuste")
        eje.set_title(nombre)
        _escribir_errores(eje, x, y, curva, malla, verdad)
    for eje in ejes.ravel()[len(familias):]:
        eje.set_visible(False)
    ejes[0, 0].legend(loc="upper left")
    figura.suptitle(titulo)
    return figura
