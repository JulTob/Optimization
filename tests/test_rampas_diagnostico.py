"""Diagnóstico: el ranking por predicción lee la estructura y la figura se dibuja."""
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Rampas.diagnostico import Diagnostico, diagnosticar, lectura_de_direcciones
from Rampas.estructuras import ESTRUCTURAS, muestra_2d
from Rampas.graficos import dibujar_diagnostico


class DiagnosticoTests(unittest.TestCase):
    def test_datos_direccionales(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 150, 0.1, 0)
        diagnostico = diagnosticar(X, y)
        case.assertEqual(diagnostico.ganador(), "Tangentes")
        grados, aporte = diagnostico.direcciones[0]
        case.assertLess(abs(grados - 30), 8)
        case.assertGreater(aporte, 0.5)

    def test_datos_radiales(case):
        X, y = muestra_2d(ESTRUCTURAS["radial"], 150, 0.1, 0)
        case.assertEqual(diagnosticar(X, y).ganador(), "Diana")

    def test_angulos_oblicuos_no_se_leen_como_interaccion_si_no_gana_tangentes(case):
        texto = lectura_de_direcciones([(57.0, 0.5), (0.0, 0.5)], "Zigurat")
        case.assertIn("no son concluyentes", texto)
        case.assertIn("ejes", lectura_de_direcciones([(2.0, 0.6), (88.0, 0.4)], "Zigurat"))

    def test_con_datos_de_prueba_externos(case):
        X, y = muestra_2d(ESTRUCTURAS["aditiva"], 120, 0.1, 1)
        X_p, y_p = muestra_2d(ESTRUCTURAS["aditiva"], 500, 0.1, 99)
        diagnostico = diagnosticar(X, y, X_p, y_p)
        case.assertEqual(set(diagnostico.errores), {"Zigurat", "Diana", "Tangentes"})
        case.assertIn("Zigurat", diagnostico._repr_markdown_())

    def test_forma_invalida(case):
        with case.assertRaises(ValueError):
            diagnosticar(np.zeros((5, 3)), np.zeros(5))

    def test_figura_con_y_sin_verdad(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 80, 0.1, 2)
        diagnostico = diagnosticar(X, y)
        for verdad in (ESTRUCTURAS["direccional"], None):
            figura = dibujar_diagnostico(diagnostico, X, y, verdad)
            case.assertEqual(len(figura.axes), 5)  #-- 4 paneles + barra de color.


if __name__ == "__main__":
    unittest.main()
