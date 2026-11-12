"""Rampas en una variable: la suma de rampas contiene a la isotónica (§2.3)."""
import unittest
from fractions import Fraction

import numpy as np
from scipy.optimize import lsq_linear

from Rampas.isotonica import a_rampas, ajustar_isotonica, evaluar_escalones, evaluar_nudos
from Rampas.rampas_1d import rampas_con_nudo_en_cada_dato, rampas_libres, rampas_regularizadas
from Rampas.senales import rampa_unitaria


def muestra():
    generador = np.random.default_rng(1)
    x = np.sort(generador.uniform(1, 6, 25))
    return x, np.sqrt(x) + generador.normal(0, 0.1, 25)


class IsotonicaTests(unittest.TestCase):
    def test_pava_es_la_proyeccion_sobre_el_cono_monotono(case):
        #-- Independiente de PAVA: θ = base + incrementos ≥ 0, resuelto como NNLS.
        x, y = muestra()
        nudos = ajustar_isotonica(x, y)
        n = len(x)
        diseno = np.tril(np.ones((n, n)))
        cotas = (np.r_[-np.inf, np.zeros(n - 1)], np.inf)
        solucion = lsq_linear(diseno, y, bounds=cotas, method="bvls", tol=1e-14)
        np.testing.assert_allclose(nudos.niveles, diseno @ solucion.x, atol=1e-9)

    def test_los_niveles_son_crecientes(case):
        nudos = ajustar_isotonica(*muestra())
        case.assertTrue(np.all(np.diff(nudos.niveles) >= 0))

    def test_empates_forman_un_nudo_con_la_media(case):
        nudos = ajustar_isotonica([1, 1, 2, 2], [0.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(nudos.posiciones, [1, 2])
        np.testing.assert_allclose(nudos.niveles, [1.0, 4.0])

    def test_escalones_coinciden_con_la_lineal_en_los_nudos(case):
        nudos = ajustar_isotonica(*muestra())
        t = nudos.posiciones
        np.testing.assert_allclose(evaluar_escalones(nudos, t), evaluar_nudos(nudos, t), atol=1e-12)
        #-- Entre dos nudos vale el nivel del nudo de la izquierda; fuera, el extremo.
        medio = (t[3] + t[4]) / 2
        case.assertEqual(evaluar_escalones(nudos, medio), nudos.niveles[3])
        case.assertEqual(evaluar_escalones(nudos, t[0] - 1), nudos.niveles[0])
        case.assertEqual(evaluar_escalones(nudos, t[-1] + 1), nudos.niveles[-1])

    def test_entradas_invalidas(case):
        with case.assertRaises(ValueError):
            ajustar_isotonica([1, 2], [1.0])
        with case.assertRaises(ValueError):
            ajustar_isotonica([1, np.nan], [1.0, 2.0])


class RampasContienenALaIsotonicaTests(unittest.TestCase):
    def test_suma_de_rampas_es_la_interpolacion_lineal(case):
        x, y = muestra()
        nudos = ajustar_isotonica(x, y)
        malla = np.linspace(0, 7, 400)
        esperado = np.interp(malla, nudos.posiciones, nudos.niveles)
        np.testing.assert_allclose(evaluar_nudos(nudos, malla), esperado, atol=1e-12)

    def test_pesos_no_negativos(case):
        _, _, _, pesos = a_rampas(ajustar_isotonica(*muestra()))
        case.assertTrue(np.all(pesos >= 0))

    def test_optimo_de_la_familia_con_nudo_en_cada_dato(case):
        x, y = muestra()
        nudos = ajustar_isotonica(x, y)
        malla = np.linspace(0, 7, 400)
        ajuste = rampas_con_nudo_en_cada_dato(x, y)
        np.testing.assert_allclose(ajuste(malla), evaluar_nudos(nudos, malla), atol=1e-8)

    def test_telescopica_exacta_en_racionales(case):
        #-- v0 + Σ (v_j − v_{j−1})·r((x − t_{j−1})/(t_j − t_{j−1})), sin error de redondeo.
        t = [Fraction(1), Fraction(3), Fraction(4), Fraction(8)]
        v = [Fraction(0), Fraction(1, 2), Fraction(2), Fraction(5)]

        def rampa_unitaria_exacta(s):
            return min(max(s, Fraction(0)), Fraction(1))

        def suma(x):
            return v[0] + sum(
                (v[j] - v[j - 1]) * rampa_unitaria_exacta((x - t[j - 1]) / (t[j] - t[j - 1]))
                for j in range(1, len(t))
                )

        for j, nivel in enumerate(v):
            case.assertEqual(suma(t[j]), nivel)
        #-- Entre dos nudos: interpolación lineal exacta.
        for j in range(1, len(t)):
            medio = (t[j - 1] + t[j]) / 2
            case.assertEqual(suma(medio), (v[j - 1] + v[j]) / 2)
        #-- Fuera del rango: constante.
        case.assertEqual(suma(Fraction(-5)), v[0])
        case.assertEqual(suma(Fraction(100)), v[-1])

    def test_rampa_unitaria_a_trozos(case):
        np.testing.assert_allclose(rampa_unitaria(np.array([-1.0, 0.0, 0.3, 1.0, 2.0])), [0, 0, 0.3, 1, 1])


class FamiliasTests(unittest.TestCase):
    def test_rampas_libres(case):
        x, y = muestra()
        nudos = ajustar_isotonica(x, y)
        error_isotonica = np.sum((y - evaluar_nudos(nudos, x)) ** 2)
        errores = []
        for K in (1, 3, 6):
            ajuste = rampas_libres(x, y, K)
            case.assertEqual(len(ajuste.pesos), K)
            case.assertTrue(np.all(ajuste.pesos >= 0))
            case.assertTrue(set(ajuste.inicios) <= set(nudos.posiciones))
            errores.append(np.sum((y - ajuste(x)) ** 2))
        #-- Más rampas nunca empeoran el ajuste, y nunca bajan de la isotónica (el óptimo monótono).
        case.assertTrue(errores[0] >= errores[1] >= errores[2] >= error_isotonica - 1e-9)

    def test_catalogo_regularizado_sin_rampas_repetidas(case):
        ajuste = rampas_regularizadas(*muestra())
        pares = set(zip(np.round(ajuste.inicios, 12), np.round(ajuste.anchos, 12)))
        case.assertEqual(len(pares), len(ajuste.inicios))


if __name__ == "__main__":
    unittest.main()
