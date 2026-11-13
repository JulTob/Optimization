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
from scipy.optimize import lsq_linear, minimize_scalar
from .dominio import Escala
from .ruido import es_blanco
from .isotonica import Nudos, ajustar_isotonica, evaluar_nudos
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


# ---------------------------------------------------------------- Pathfinder
def ajustar_pathfinder(X, y, pasos=None, banda=2.0, n_barrido=45):
    """Pathfinder (ascenso de rampas): avanzar y escalar, una rampa por paso.

    Desde el origen del dominio normalizado, cada paso hace cuatro cosas:

    1. Elige la dirección de mejor crecimiento isotónico del residuo,
       como en Tangentes (el isotónico ve toda la estructura).
    2. Comprueba la BLANCURA del residuo por delante de la posición,
       ordenado en esa dirección: si es ruido blanco (autocorrelación a
       retardo 1 dentro de la banda banda/√n; Rampas.ruido), no queda
       estructura que escalar y el camino termina. Es el mismo criterio
       de parada que rampas por división.
    3. Si queda color, el paso es el PRIMER TRAMO DE SUBIDA del
       isotónico por delante: el primer nodo del camino. El pathfinder
       puede CAMINAR por el llano (de la posición hasta a, sin subir)
       antes de ESCALAR (de a a b).
    4. Reajusta la base y todos los pesos juntos (mínimos cuadrados con
       pesos ≥ 0, §2.3) y avanza la posición al punto de saturación.

    Cada rampa nueva vale 0 por debajo de la posición desde la que se
    lanzó (componente a componente), así que lo ya recorrido no se
    toca. `pasos` limita los pasos (None: sin tope; el camino acaba
    igual, porque avanza en cada paso y los nudos son los datos,
    finitos).

    El modelo queda descrito por la base y los tramos (û_k, a_k, b_k)
    con sus pesos: la rampa k vale w_k · r((x·û_k − a_k)/(b_k − a_k)).
    El camino se recupera con `camino_del_pathfinder`.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("y debe ser un vector finito no vacío.")
    if pasos is not None and (not isinstance(pasos, (int, np.integer)) or pasos < 1):
        raise ValueError("pasos debe ser un entero mayor o igual que 1, o None.")
    banda = float(banda)
    if not np.isfinite(banda):
        raise ValueError("banda debe ser finita.")

    escala = Escala.de_los_datos(X)
    Z = escala(X)
    if len(Z) != len(y):
        raise ValueError("X e y deben tener el mismo número de datos.")

    posicion = np.zeros(2)
    tramos = []         #-- (ángulo, inicio, fin) de cada rampa aceptada, en orden.
    pesos = np.zeros(0)
    base = float(np.mean(y))

    while pasos is None or len(tramos) < pasos:
        residuo = y - _evaluar_pathfinder(Z, base, tramos, pesos)
        angulo = mejor_angulo(Z, residuo, n_barrido)
        rumbo = _rumbo(angulo)
        proyectados = Z @ rumbo
        por_delante = proyectados > float(posicion @ rumbo) + 1e-12
        if not np.any(por_delante):
            break
            #-- Ningún dato por delante de la posición: fin del camino.
        orden = np.argsort(proyectados[por_delante])
        if es_blanco(residuo[por_delante][orden], banda):
            break
            #-- Blanco incluso en la mejor dirección: no queda estructura.
        tramo = _primera_subida(proyectados[por_delante], residuo[por_delante])
        if tramo is None:
            break
            #-- El isotónico de delante es plano: nada que escalar.
        tramos = tramos + [(float(angulo),) + tramo]
        base, pesos = _pesos_del_pathfinder(Z, y, tramos)
        posicion = posicion + (tramo[1] - float(posicion @ rumbo)) * rumbo
        #-- La posición salta al punto de saturación de la rampa nueva.

    return Modelo2D("Pathfinder", escala, _componentes_del_pathfinder(base, tramos, pesos))


def camino_del_pathfinder(modelo):
    """Los puntos del camino del Pathfinder (llanos y subidas), en el
    dominio normalizado.

    El camino es en sí mismo un diagnóstico visual: recto en datos
    direccionales, por la diagonal en radiales, en escalera (tramos a
    0° y 90°) en aditivos.
    """
    posicion = np.zeros(2)
    puntos = [posicion.copy()]
    for componente in modelo.componentes:
        angulo = componente.proyeccion.angulo
        if angulo is None or len(componente.nudos.posiciones) < 2:
            continue
        inicio, fin = componente.nudos.posiciones[0], componente.nudos.posiciones[-1]
        avance = float(inicio) - float(posicion @ _rumbo(angulo))
        if avance > 1e-12:
            posicion = posicion + avance * _rumbo(angulo)
            puntos.append(posicion.copy())
            #-- Tramo llano: caminar hasta donde empieza la subida.
        posicion = posicion + float(fin - inicio) * _rumbo(angulo)
        puntos.append(posicion.copy())
    return np.array(puntos)


def _rumbo(angulo):
    """El vector unitario û = (cos θ, sen θ): θ en [0°, 90°] ⇒ û ≥ 0."""
    return np.array([np.cos(angulo), np.sin(angulo)])


def _primera_subida(proyectados, residuo):
    """El primer tramo [a, b] donde el isotónico del residuo sube.

    Es el «primer nodo» del camino: el paso más cercano, no el mejor
    posible. El paso glotón saltaría lejos y dejaría atrás,
    irrecuperable, lo no explicado; el paso mínimo deja que el camino
    se adapte, como en el ascenso por gradiente.
    """
    nudos = ajustar_isotonica(proyectados, residuo)
    sube = np.diff(nudos.niveles) > 1e-12
    arranques = np.nonzero(sube)[0]
    if arranques.size == 0:
        return None
    inicio = int(arranques[0])
    fin = inicio + 1
    while fin < len(sube) and sube[fin]:
        fin += 1
        #-- La subida sigue mientras los niveles crezcan seguidos.
    return (float(nudos.posiciones[inicio]), float(nudos.posiciones[fin]))


def _rampa_del_tramo(Z, tramo):
    angulo, inicio, fin = tramo
    return np.clip((Z @ _rumbo(angulo) - inicio) / (fin - inicio), 0.0, 1.0)


def _pesos_del_pathfinder(Z, y, tramos):
    """Reajuste conjunto: base libre y un peso ≥ 0 por rampa."""
    diseno = np.column_stack(
            [np.ones(len(Z))] + [_rampa_del_tramo(Z, tramo) for tramo in tramos]
            )
    resultado = lsq_linear(
            diseno,
            y,
            bounds=(np.r_[-np.inf, np.zeros(len(tramos))], np.inf),
            method="bvls",
            tol=1e-12,
            )
    if not resultado.success:
        raise RuntimeError(f"El ajuste no ha convergido: {resultado.message}")
    return float(resultado.x[0]), resultado.x[1:]


def _evaluar_pathfinder(Z, base, tramos, pesos):
    total = np.full(len(Z), base)
    for tramo, peso in zip(tramos, pesos):
        total = total + peso * _rampa_del_tramo(Z, tramo)
    return total


def _componentes_del_pathfinder(base, tramos, pesos):
    """Cada paso es una Componente de dos nudos; la base va en el primero."""
    if not tramos:
        return [Componente(eje(0), Nudos(np.zeros(1), np.array([base])))]
    componentes = []
    for k, ((angulo, inicio, fin), peso) in enumerate(zip(tramos, pesos)):
        nivel = base if k == 0 else 0.0
        componentes.append(Componente(
                direccion(angulo),
                Nudos(np.array([inicio, fin]), np.array([nivel, nivel + peso])),
                ))
    return componentes
