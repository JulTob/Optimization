"""Diagnóstico: ¿cómo se combinan las dos variables?

Se ajustan los tres modelos y se ordenan por ERROR DE PREDICCIÓN
(datos que el ajuste no ha visto). El ganador indica la estructura:

    Zigurat    aditiva     cada variable actúa por separado.
    Diana      radial      solo importa la distancia al origen.
    Tangentes  direccional las variables actúan juntas, en una o varias direcciones.

Las direcciones de Tangentes solo APOYAN la lectura: se interpretan si gana
Tangentes o si caen en los ejes. Cuentan las que aportan al menos un 10 %
de la subida total. Con datos aditivos o radiales, Tangentes suele usar
también direcciones oblicuas, así que ahí no se leen como interacción.
"""
import numpy as np
from dataclasses import dataclass
from .modelos_2d import ajustar_diana, ajustar_tangentes, ajustar_zigurat

AJUSTES = {
    "Zigurat": ajustar_zigurat,
    "Diana": ajustar_diana,
    "Tangentes": ajustar_tangentes,
    }
ESTRUCTURA = {
    "Zigurat": "aditiva: cada variable actúa por separado",
    "Diana": "radial: solo importa la distancia al origen",
    "Tangentes": "direccional: las variables actúan juntas",
    }
APORTE_MINIMO = 0.10
EMPATE = 0.05
TOLERANCIA_EJE = 10


@dataclass(frozen=True)
class Diagnostico:
    modelos: dict           #-- nombre → Modelo2D, ajustado con todos los datos.
    errores: dict           #-- nombre → error cuadrático medio de predicción.
    direcciones: list       #-- (grados, aporte) de Tangentes, de mayor a menor aporte.

    def orden(self):
        return sorted(self.errores, key=self.errores.get)

    def ganador(self):
        return self.orden()[0]

    def hay_empate(self):
        primero, segundo = self.orden()[:2]
        return self.errores[segundo] <= (1 + EMPATE) * self.errores[primero]

    def lectura(self):
        primero, segundo = self.orden()[:2]
        frases = [f"Mejor predicción: {primero}. Estructura {ESTRUCTURA[primero]}."]
        if self.hay_empate():
            frases.append(f"Empate técnico con {segundo} (menos de un {EMPATE:.0%} de diferencia).")
        frases.append(lectura_de_direcciones(self.direcciones, primero))
        return " ".join(frases)

    def __str__(self):
        filas = [f"  {nombre:<10} {self.errores[nombre]:.4f}" for nombre in self.orden()]
        return "Error de predicción (ECM):\n" + "\n".join(filas) + "\n" + self.lectura()

    def _repr_markdown_(self):
        #-- Colab muestra esto al dejar el objeto al final de una celda.
        filas = [
            f"| {'**' + n + '**' if n == self.ganador() else n} | {self.errores[n]:.4f} |"
            for n in self.orden()
            ]
        tabla = "| Modelo | ECM de predicción |\n|---|---|\n" + "\n".join(filas)
        return tabla + "\n\n" + self.lectura()


def lectura_de_direcciones(direcciones, ganador):
    """Qué dicen los ángulos relevantes de Tangentes. El ranking manda; los ángulos apoyan."""
    grados = [g for g, _ in direcciones]
    lista = ", ".join(f"{g:.0f}°" for g in grados)
    en_ejes = all(min(g, 90 - g) <= TOLERANCIA_EJE for g in grados)
    if grados and en_ejes:
        return f"Las direcciones de Tangentes caen en los ejes ({lista}): apoyan la aditividad."
    if ganador == "Tangentes" and len(grados) == 1:
        return f"Una sola dirección relevante ({lista}): las variables actúan como una combinación lineal."
    if ganador == "Tangentes":
        return f"Varias direcciones oblicuas ({lista}): hay interacción entre las variables."
    return f"Las direcciones de Tangentes ({lista or 'ninguna'}) no son concluyentes aquí."


def direcciones_relevantes(modelo_tangentes):
    aportes = [(np.degrees(c.proyeccion.angulo), c.amplitud()) for c in modelo_tangentes.componentes]
    total = sum(a for _, a in aportes)
    if total <= 0:
        return []
    relevantes = [(g, a / total) for g, a in aportes if a / total >= APORTE_MINIMO]
    return sorted(relevantes, key=lambda par: -par[1])


def separar(n, fraccion_prueba, semilla):
    orden = np.random.default_rng(semilla).permutation(n)
    corte = int(n * fraccion_prueba)
    return orden[corte:], orden[:corte]


def diagnosticar(X, y, X_prueba=None, y_prueba=None, fraccion_prueba=0.25, semilla=0):
    """Ajusta los tres modelos, mide su error de predicción y lee la estructura.

    Sin datos de prueba, se reserva una parte de los datos para medir el error;
    después cada modelo se reajusta con todos los datos para representarlo.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) != len(y):
        raise ValueError("X debe tener forma (n, 2) e y forma (n,).")
    if X_prueba is None:
        entrenamiento, prueba = separar(len(X), fraccion_prueba, semilla)
        X_ajuste, y_ajuste = X[entrenamiento], y[entrenamiento]
        X_prueba, y_prueba = X[prueba], y[prueba]
    else:
        X_ajuste, y_ajuste = X, y
    errores = {
        nombre: float(np.mean((y_prueba - ajustar(X_ajuste, y_ajuste).predecir(X_prueba)) ** 2))
        for nombre, ajustar in AJUSTES.items()
        }
    modelos = {nombre: ajustar(X, y) for nombre, ajustar in AJUSTES.items()}
    return Diagnostico(modelos, errores, direcciones_relevantes(modelos["Tangentes"]))
