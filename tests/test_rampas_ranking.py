"""Pruebas de los modelos de referencia y del ranking (§3.2)."""

import unittest

import numpy as np

from Rampas.referencias import (
    ajustar_logistica,
    ajustar_polinomio,
    ajustar_potencia,
    catalogo_de_referencias,
)
from Rampas.ranking import catalogo_de_rampas, muestra_1d, ranking_de_modelos


class Referencias(unittest.TestCase):

    def test_recta_exacta(case):
        x = np.linspace(0.0, 1.0, 10)
        y = 2.0 + 3.0 * x
        f = ajustar_polinomio(x, y, grado=1)
        np.testing.assert_allclose(f(x), y, atol=1e-9)

    def test_cubico_exacto(case):
        x = np.linspace(-1.0, 1.0, 12)
        y = 1.0 - x + 0.5 * x**3
        f = ajustar_polinomio(x, y, grado=3)
        np.testing.assert_allclose(f(x), y, atol=1e-9)

    def test_potencia_exacta(case):
        x = np.linspace(1.0, 6.0, 15)
        y = 2.0 * x**0.5
        f = ajustar_potencia(x, y)
        np.testing.assert_allclose(f(x), y, atol=1e-6)

    def test_potencia_exige_x_positivo(case):
        with case.assertRaises(ValueError):
            ajustar_potencia([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    def test_logistica_recupera_senal_limpia(case):
        from scipy.special import expit
        x = np.linspace(0.0, 1.0, 40)
        y = 0.1 + 0.9 * expit(10.0 * (x - 0.4))
        f = ajustar_logistica(x, y)
        np.testing.assert_allclose(f(x), y, atol=1e-6)

    def test_catalogo_cinco_modelos(case):
        case.assertEqual(len(catalogo_de_referencias()), 5)

    def test_entradas_invalidas(case):
        with case.assertRaises(ValueError):
            ajustar_polinomio([1.0], [1.0], grado=1)
        with case.assertRaises(ValueError):
            ajustar_polinomio([1.0, 2.0, np.nan], [1.0, 2.0, 3.0], grado=1)


class Ranking(unittest.TestCase):

    def modelos_rapidos(case):
        catalogo = catalogo_de_rampas()
        return {
            "Isotónica lineal": catalogo["Isotónica lineal"],
            "Isotónica escalonada": catalogo["Isotónica escalonada"],
            "Recta": catalogo_de_referencias()["Recta"],
        }

    def test_tabla_completa_y_ordenada(case):
        tabla = ranking_de_modelos(
            np.sqrt, 1.0, 6.0, sigma=0.1,
            n=15, n_ajustes=2, n_pruebas=2, n_malla=50,
            semilla=1, modelos=case.modelos_rapidos(),
            )
        case.assertEqual(len(tabla), 3)
        case.assertEqual(
            list(tabla.columns),
            [ "Test (media)", "Test (varianza)",
              "Señal (media)", "Señal (varianza)",
              "Ajustes fallidos" ],
            )
        errores = tabla["Señal (media)"].to_numpy()
        case.assertTrue(np.all(np.diff(errores) >= 0))
        case.assertTrue(np.all(tabla["Ajustes fallidos"] == 0))

    def test_misma_semilla_misma_tabla(case):
        def correr():
            return ranking_de_modelos(
                np.sqrt, 1.0, 6.0, sigma=0.1,
                n=15, n_ajustes=2, n_pruebas=2, n_malla=50,
                semilla=7, modelos=case.modelos_rapidos(),
                )
        primera = correr()
        segunda = correr()
        case.assertTrue(primera.equals(segunda))

    def test_modelo_que_falla_se_cuenta(case):
        def siempre_falla(x, y):
            raise ValueError("ajuste imposible")

        tabla = ranking_de_modelos(
            np.sqrt, 1.0, 6.0, sigma=0.1,
            n=15, n_ajustes=2, n_pruebas=2, n_malla=50,
            semilla=1,
            modelos={ "Roto": siempre_falla,
                      "Recta": catalogo_de_referencias()["Recta"] },
            )
        case.assertEqual(int(tabla.loc["Roto", "Ajustes fallidos"]), 2)
        case.assertTrue(np.isnan(tabla.loc["Roto", "Señal (media)"]))
        case.assertEqual(tabla.index[-1], "Roto")
        #-- Los NaN van al final del ranking.

    def test_muestra_usa_el_dominio(case):
        generador = np.random.default_rng(0)
        x, y = muestra_1d(np.sqrt, 1.0, 6.0, sigma=0.0, n=50, generador=generador)
        case.assertTrue(np.all((1.0 <= x) & (x <= 6.0)))
        np.testing.assert_allclose(y, np.sqrt(x))
        case.assertTrue(np.all(np.diff(x) >= 0))

    def test_entradas_invalidas(case):
        with case.assertRaises(ValueError):
            ranking_de_modelos(np.sqrt, 6.0, 1.0, sigma=0.1)
        with case.assertRaises(ValueError):
            ranking_de_modelos(np.sqrt, 1.0, 6.0, sigma=-1.0)
        with case.assertRaises(ValueError):
            ranking_de_modelos(np.sqrt, 1.0, 6.0, sigma=0.1, n_ajustes=1)
        with case.assertRaises(ValueError):
            ranking_de_modelos(np.sqrt, 1.0, 6.0, sigma=0.1, modelos={})


if __name__ == "__main__":
    unittest.main()
