"""Zigurat, Diana y Tangentes: monotonía garantizada y diagnóstico por error de predicción."""
import unittest

import numpy as np

from Rampas.estructuras import ANGULO_VERDADERO, ESTRUCTURAS, muestra_2d
from Rampas.experimento import EQUIVALENTE, una_repeticion
from Rampas.modelos_2d import (
        ajustar_pathfinder, ajustar_diana, ajustar_tangentes, ajustar_zigurat,
        camino_del_pathfinder,
        )


class MonotoniaTests(unittest.TestCase):
    def test_los_tres_modelos_son_crecientes_en_el_cuadrante(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 120, 0.1, 0)
        lado = np.linspace(0, 1, 40)
        malla = np.stack(np.meshgrid(lado, lado), -1)
        for ajustar in (ajustar_zigurat, ajustar_diana, ajustar_tangentes, ajustar_pathfinder):
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
    def test_pathfinder_recupera_la_direccion(case):
        X, y = muestra_2d(ESTRUCTURAS["direccional"], 150, 0.1, 0)
        angulos = ajustar_pathfinder(X, y).angulos_grados()
        case.assertTrue(any(abs(a - np.degrees(ANGULO_VERDADERO)) <= 8 for a in angulos))

    def test_el_camino_del_pathfinder_es_creciente(case):
        X, y = muestra_2d(ESTRUCTURAS["radial"], 150, 0.1, 0)
        camino = camino_del_pathfinder(ajustar_pathfinder(X, y))
        np.testing.assert_array_equal(camino[0], [0.0, 0.0])
        case.assertGreaterEqual(np.diff(camino, axis=0).min(), -1e-9)
            #-- Cada tramo avanza: nunca se retrocede en ninguna coordenada.

    def test_pathfinder_con_datos_planos_da_una_constante(case):
        generador = np.random.default_rng(0)
        X = generador.uniform(0, 1, (40, 2))
        y = np.full(40, 3.0)
        modelo = ajustar_pathfinder(X, y)
        np.testing.assert_allclose(modelo.predecir(X), 3.0)
        case.assertEqual(len(camino_del_pathfinder(modelo)), 1)
            #-- Sin nada que escalar, el camino se queda en el origen.

    def test_pathfinder_rechaza_entradas_invalidas(case):
        X, y = muestra_2d(ESTRUCTURAS["radial"], 30, 0.1, 0)
        with case.assertRaises(ValueError):
            ajustar_pathfinder(X, y, pasos=0)
        with case.assertRaises(ValueError):
            ajustar_pathfinder(X, y, umbral=-0.1)
        with case.assertRaises(ValueError):
            ajustar_pathfinder(X, y[:-1])

    def test_la_red_de_zigurat_son_rectas_en_los_ejes(case):
        from Rampas.graficos import dibujar_redes, nudos_activos
        X, z = muestra_2d(ESTRUCTURAS["aditiva"], 80, 0.1, 0)
        modelo = ajustar_zigurat(X, z)
        for componente in modelo.componentes:
            nudos = nudos_activos(componente)
            case.assertTrue(nudos.size >= 2)
            case.assertTrue(np.all((0 <= nudos) & (nudos <= 1)))
        muestras = {"aditiva": (X, z, ESTRUCTURAS["aditiva"])}
        figura = dibujar_redes(muestras, resolucion=40)
        case.assertEqual(len(figura.axes), 3)
        from Rampas.graficos import dibujar_datos_2d
        figura = dibujar_datos_2d(muestras, resolucion=40)
        case.assertIn("n = 80", figura.axes[0].get_title())

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
                ganadores.append(min(EQUIVALENTE.values(), key=errores.get))
                    #-- El concurso es entre las tres hipótesis; el Pathfinder no diagnostica.
            with case.subTest(estructura=estructura):
                case.assertGreaterEqual(ganadores.count(EQUIVALENTE[estructura]), 2)


if __name__ == "__main__":
    unittest.main()
