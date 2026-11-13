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
        eje.set(title=nombre, xlabel="x")
    np.atleast_1d(ejes)[0].set_ylabel("y")
    figura.colorbar(relleno, ax=ejes)
    return figura


def dibujar_caminos(muestras, titulo=""):
    """El camino del Pathfinder sobre la señal, un panel por estructura.

    `muestras` es un diccionario nombre → (X, z, verdad). Cada panel
    dibuja los contornos de la señal verdadera, los datos y el camino
    P₀, P₁, … del Pathfinder ajustado a esos datos: la forma del camino
    es el diagnóstico.
    """
    from .modelos_2d import ajustar_pathfinder, camino_del_pathfinder
    lado = np.linspace(0, 1, 80)
    A, B = np.meshgrid(lado, lado)
    puntos = np.column_stack([A.ravel(), B.ravel()])
    figura, ejes = plt.subplots(1, len(muestras), figsize=(4 * len(muestras), 3.8), sharey=True)
    for eje, (nombre, (X, z, verdad)) in zip(np.atleast_1d(ejes), muestras.items()):
        eje.contourf(A, B, verdad(puntos).reshape(A.shape), levels=12, alpha=0.8)
        eje.scatter(X[:, 0], X[:, 1], c="white", s=5, alpha=0.6)
        camino = camino_del_pathfinder(ajustar_pathfinder(X, z))
        eje.plot(camino[:, 0], camino[:, 1], "-o", color="tab:red", lw=2, ms=3)
        eje.set(title=nombre, xlabel="x", xlim=(0, 1), ylim=(0, 1))
    np.atleast_1d(ejes)[0].set_ylabel("y")
    if titulo:
        figura.suptitle(titulo)
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
    """Errores medios en la esquina inferior derecha.

    ⟨|y − f(x)|²⟩     media sobre los datos y (entrenamiento);
    ⟨|f(x) − s(x)|²⟩  media frente a la señal generadora s, en todo el rango (si se conoce).
    """
    lineas = [rf"$\langle |y-f(x)|^2 \rangle$ = {np.mean((y - curva(x)) ** 2):.4f}"]
    if verdad is not None:
        lineas.append(rf"$\langle |f(x)-s(x)|^2 \rangle$ = {np.mean((verdad(malla) - curva(malla)) ** 2):.4f}")
    eje.text(
            0.97, 0.04, "\n".join(lineas),
            transform=eje.transAxes, ha="right", va="bottom", fontsize=9,
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
    familias[f"Rampas por división (K = {len(por_division.pesos)}, emergente)"] = por_division
    malla = np.linspace(np.min(x), np.max(x), 1000)
    figura, ejes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True)
    for eje, (nombre, curva) in zip(ejes.ravel(), familias.items()):
        eje.scatter(x, y, color="black", s=14, zorder=3, label="$y$ (datos)")
        if verdad is not None:
            eje.plot(malla, verdad(malla), "--", color="gray", label="$s(x)$ (señal)")
        eje.plot(malla, curva(malla), color="tab:red", lw=2, label="$f(x)$ (ajuste)")
        eje.set_title(nombre)
        _escribir_errores(eje, x, y, curva, malla, verdad)
    for eje in ejes.ravel()[len(familias):]:
        eje.set_visible(False)
    ejes[0, 0].legend(loc="upper left")
    figura.suptitle(titulo)
    return figura


def nudos_activos(componente, tolerancia=1e-9):
    """Los nudos donde una rampa de la componente empieza o acaba.

    Entre dos nudos consecutivos hay una rampa de peso igual al salto de
    niveles; si el salto es nulo, la rampa no existe y sus nudos no
    cambian nada. En el plano, cada nudo activo es una curva de cambio de
    pendiente: recta, círculo o recta oblicua según la proyección.
    """
    posiciones = np.asarray(componente.nudos.posiciones, dtype=float)
    saltos = np.diff(np.asarray(componente.nudos.niveles, dtype=float))
    if saltos.size == 0:
        return np.zeros(0)
    escala = max(float(np.max(np.abs(saltos))), 1e-300)
    activos = np.nonzero(saltos > tolerancia * escala)[0]
    return np.unique(np.concatenate([posiciones[activos], posiciones[activos + 1]]))


def dibujar_redes(muestras, resolucion=200):
    """La red de nudos de cada modelo en cada estructura: el 3×3 a la vista.

    `muestras` es un diccionario nombre → (X, z, verdad). Filas:
    estructuras; columnas: Zigurat, Diana y Tangentes ajustados a esos
    datos. Cada modelo es lineal a trozos sobre la red que forman sus
    nudos: cuadrícula (Zigurat), anillos (Diana) o familias de rectas
    paralelas, una por dirección (Tangentes, un color por familia).
    """
    from .modelos_2d import ajustar_diana, ajustar_tangentes, ajustar_zigurat
    ajustes = (ajustar_zigurat, ajustar_diana, ajustar_tangentes)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lado = np.linspace(0, 1, resolucion)
    A, B = np.meshgrid(lado, lado)
    puntos = np.column_stack([A.ravel(), B.ravel()])

    figura, ejes = plt.subplots(
            len(muestras), len(ajustes),
            figsize=(3.6 * len(ajustes), 3.4 * len(muestras)),
            sharex=True, sharey=True, squeeze=False,
            )
    for fila, (nombre, (X, z, verdad)) in enumerate(muestras.items()):
        for columna, ajustar in enumerate(ajustes):
            eje = ejes[fila, columna]
            modelo = ajustar(X, z)
            eje.contourf(A, B, verdad(puntos).reshape(A.shape), levels=12, cmap="Greys", alpha=0.35)
            for k, componente in enumerate(modelo.componentes):
                nudos = nudos_activos(componente)
                if nudos.size == 0:
                    continue
                proyectados = componente.proyeccion(modelo.escala(puntos)).reshape(A.shape)
                eje.contour(A, B, proyectados, levels=nudos,
                            colors=colores[k % len(colores)], linewidths=0.8)
            eje.set(xlim=(0, 1), ylim=(0, 1))
            if fila == 0:
                eje.set_title(modelo.nombre)
            if columna == 0:
                eje.set_ylabel(f"{nombre}\ny")
            if fila == len(muestras) - 1:
                eje.set_xlabel("x")
    return figura
