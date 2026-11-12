"""Zigurat, Diana y Tangentes: monotonía garantizada y diagnóstico por error de predicción."""
import unittest

import numpy as np

from Rampas.estructuras import ANGULO_VERDADERO, ESTRUCTURAS, muestra_2d
from Rampas.experimento import EQUIVALENTE, una_repeticion
from Rampas.modelos_2d import ajustar_diana, ajustar_tangentes, ajustar_zigurat


class MonotoniaTests(unittest.TestCase):
    def test_los_tres_modelos_son_crecientes_en_el_cuadrante(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 120, 0.1, 0)
        lado = np.linspace(0, 1, 40)
        malla = np.stack(np.meshgrid(lado, lado), -1)
        for ajustar in (ajustar_zigurat, ajustar_diana, ajustar_tangentes):
            modelo = ajustar(X, y)
            puntos = modelo.escala.minimo + modelo.escala.rango * malla
            valores = modelo.predecir(puntos.reshape(-1, 2)).reshape(40, 40)
            with case.subTest(modelo=modelo.nombre):
                case.assertGreaterEqual(np.diff(valores, axis=0).min(), -1e-9)
                case.assertGreaterEqual(np.diff(valores, axis=1).min(), -1e-9)

    def test_variable_constante_se_rechaza(case):
        X = np.column_stack([np.ones(10), np.linspace(0, 1, 10)])
        with case.assertRaises(ValueError):
            ajustar_zigurat(X, np.arange(10.0))


class DiagnosticoTests(unittest.TestCase):
    def test_tangentes_recupera_la_direccion(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 150, 0.1, 0)
        angulos = ajustar_tangentes(X, y).angulos_grados()
        case.assertTrue(any(abs(a - np.degrees(ANGULO_VERDADERO)) <= 8 for a in angulos))

    def test_gana_el_modelo_de_la_estructura_real(case):
        #-- Tres semillas por estructura; la diagonal debe ganar en la mayoría.
        for estructura, verdad in ESTRUCTURAS.items():
            ganadores = []
            for semilla in range(3):
                errores, _ = una_repeticion(verdad, semilla)
                ganadores.append(min(errores, key=errores.get))
            with case.subTest(estructura=estructura):
                case.assertGreaterEqual(ganadores.count(EQUIVALENTE[estructura]), 2)


if __name__ == "__main__":
    unittest.main()
