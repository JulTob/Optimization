"""Pruebas del estimador de ruido y del umbral de mejora."""

import unittest

import numpy as np

from Rampas.ruido import umbral_de_mejora, varianza_del_ruido


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


if __name__ == "__main__":
    unittest.main()
