"""Pruebas del estimador de ruido y del umbral de mejora."""

import unittest

import numpy as np

from Rampas.ruido import (
    autocorrelacion_primera,
    calibrar_umbral,
    diferencias_de_ruido,
    es_blanco,
    umbral_de_mejora,
    varianza_del_ruido,
)


class Ruido(unittest.TestCase):

    def test_recupera_la_varianza_en_1d(case):
        generador = np.random.default_rng(0)
        x = np.sort(generador.uniform(0, 1, 2000))
        y = np.sqrt(x) + generador.normal(0, 0.1, 2000)
        estimada = varianza_del_ruido(x, y)
        case.assertAlmostEqual(estimada, 0.01, delta=0.002)

    def test_un_vector_equivale_a_una_columna(case):
        generador = np.random.default_rng(1)
        x = np.sort(generador.uniform(0, 1, 100))
        y = x + generador.normal(0, 0.1, 100)
        case.assertAlmostEqual(
            varianza_del_ruido(x, y),
            varianza_del_ruido(x[:, None], y),
            places=12,
            )

    def test_recupera_la_varianza_en_2d(case):
        generador = np.random.default_rng(0)
        Z = generador.uniform(0, 1, (2000, 2))
        y = Z[:, 0] + Z[:, 1] + generador.normal(0, 0.1, 2000)
        estimada = varianza_del_ruido(Z, y)
        case.assertAlmostEqual(estimada, 0.01, delta=0.003)

    def test_sin_ruido_la_varianza_es_casi_cero(case):
        x = np.linspace(0, 1, 200)
        case.assertLess(varianza_del_ruido(x, np.full(200, 2.0)), 1e-12)

    def test_umbral_al_95_es_chi_cuadrado(case):
        case.assertAlmostEqual(umbral_de_mejora(1.0, 0.95), 3.841, delta=0.001)
        case.assertAlmostEqual(umbral_de_mejora(0.01, 0.95), 0.03841, delta=1e-5)

    def test_mas_confianza_sube_el_umbral(case):
        case.assertLess(umbral_de_mejora(1.0, 0.5), umbral_de_mejora(1.0, 0.95))
        case.assertLess(umbral_de_mejora(1.0, 0.95), umbral_de_mejora(1.0, 0.999))

    def test_las_diferencias_estan_centradas(case):
        generador = np.random.default_rng(2)
        x = np.sort(generador.uniform(0, 1, 300))
        y = x**2 + generador.standard_t(df=3, size=300) * 0.1
        #-- Ruido de colas pesadas, nada gaussiano: el estimador no lo supone.
        muestras = diferencias_de_ruido(x, y)
        case.assertEqual(len(muestras), 300)
        case.assertAlmostEqual(float(muestras.mean()), 0.0, places=12)

    def test_calibrar_umbral_es_determinista_y_crece_con_la_confianza(case):
        generador = np.random.default_rng(3)
        x = np.sort(generador.uniform(0, 1, 60))
        y = x + generador.normal(0, 0.1, 60)

        def robo(entradas, ruido):
            return float(np.max(ruido) ** 2)

        bajo = calibrar_umbral(x, y, robo, confianza=0.5)
        alto = calibrar_umbral(x, y, robo, confianza=0.99)
        case.assertEqual(bajo, calibrar_umbral(x, y, robo, confianza=0.5))
        case.assertLessEqual(bajo, alto)

    def test_el_ruido_puro_es_blanco(case):
        generador = np.random.default_rng(5)
        case.assertTrue(es_blanco(generador.standard_t(df=3, size=200)))
        #-- Blanco aunque las colas sean pesadas: la banda no supone gaussiana.

    def test_un_residuo_con_estructura_tiene_color(case):
        x = np.linspace(0, 1, 100)
        residuo = np.sin(2 * np.pi * x)
        #-- Estructura que una recta no capturó: vecinos parecidos, rho_1 alta.
        case.assertFalse(es_blanco(residuo))
        case.assertGreater(autocorrelacion_primera(residuo), 0.9)

    def test_el_sobreajuste_alterna_y_tambien_es_blanco(case):
        alterno = np.array([1.0, -1.0] * 50)
        case.assertLess(autocorrelacion_primera(alterno), 0.0)
        case.assertTrue(es_blanco(alterno))
        #-- rho_1 negativa no pide más rampas: la prueba es de un solo lado.

    def test_pocos_residuos_se_declaran_blancos(case):
        case.assertTrue(es_blanco([1.0, 5.0, 2.0]))

    def test_entradas_invalidas(case):
        with case.assertRaises(ValueError):
            umbral_de_mejora(1.0, confianza=0.0)
        with case.assertRaises(ValueError):
            umbral_de_mejora(1.0, confianza=1.0)
        with case.assertRaises(ValueError):
            umbral_de_mejora(-1.0)
        with case.assertRaises(ValueError):
            varianza_del_ruido([1.0], [1.0])
        with case.assertRaises(ValueError):
            varianza_del_ruido(np.ones((3, 2)), np.ones(2))
        with case.assertRaises(ValueError):
            calibrar_umbral(np.ones(5), np.ones(5), lambda X, r: 0.0, confianza=1.0)
        with case.assertRaises(ValueError):
            es_blanco(np.ones(10), banda=float("inf"))
        with case.assertRaises(ValueError):
            autocorrelacion_primera([1.0])


if __name__ == "__main__":
    unittest.main()
