"""Rampas en una variable: la suma de rampas contiene a la isotónica (§2.3)."""
import unittest
from fractions import Fraction

import numpy as np
from scipy.optimize import lsq_linear

from Rampas.isotonica import a_rampas, ajustar_isotonica, evaluar_escalones, evaluar_nudos
from Rampas.rampas_1d import rampas_con_nudo_en_cada_dato, rampas_libres, rampas_por_division, rampas_regularizadas
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

    def test_rampas_por_division(case):
        x, y = muestra()
        nudos = ajustar_isotonica(x, y)
        error_isotonica = np.sum((y - evaluar_nudos(nudos, x)) ** 2)

        #-- Más profundidad nunca empeora el entrenamiento, y nunca baja de la isotónica.
        #-- Con confianza mínima el umbral es ~0 y la división llega lo más lejos posible.
        errores = [
            np.sum((y - rampas_por_division(x, y, profundidad=k, confianza=1e-9)(x)) ** 2)
            for k in (1, 2, 4)
        ]
        case.assertTrue(all(a >= b - 1e-9 for a, b in zip(errores, errores[1:])))
        case.assertTrue(errores[-1] >= error_isotonica - 1e-9)

        #-- Profundidad k: como mucho 2^k − 1 cortes, o sea 2^k rampas.
        for k in (1, 2, 4):
            ajuste = rampas_por_division(x, y, profundidad=k, confianza=1e-9)
            case.assertTrue(1 <= len(ajuste.pesos) <= 2 ** k)

        #-- Las rampas son contiguas: cada una empieza donde acaba la anterior.
        ajuste = rampas_por_division(x, y, profundidad=3)
        np.testing.assert_allclose(ajuste.inicios[1:], ajuste.inicios[:-1] + ajuste.anchos[:-1])

        #-- Sin tope de profundidad, K emerge y se para antes de cortar todos los puntos.
        elegido = rampas_por_division(x, y)
        case.assertTrue(1 <= len(elegido.pesos) < len(nudos.posiciones))

        #-- Más confianza exige mejoras mayores: nunca salen más rampas.
        exigente = rampas_por_division(x, y, confianza=1 - 1e-12)
        case.assertTrue(len(exigente.pesos) <= len(elegido.pesos))

        #-- La confianza debe estar entre 0 y 1 (abiertos).
        with case.assertRaises(ValueError):
            rampas_por_division(x, y, confianza=1.0)

        #-- Determinista: dos llamadas idénticas, mismo resultado.
        np.testing.assert_array_equal(
            rampas_por_division(x, y).pesos,
            rampas_por_division(x, y).pesos,
            )

    def test_figura_de_familias_con_errores_legibles(case):
        import matplotlib
        matplotlib.use("Agg")
        from Rampas.graficos import dibujar_familias
        x, y = muestra()
        figura = dibujar_familias(x, y, verdad=np.sqrt)
        textos = [t.get_text() for eje in figura.axes for t in eje.texts]
        case.assertEqual(len(textos), 6)
        for texto in textos:
            case.assertIn("s(x)", texto)
            case.assertLess(len(texto), 120)
            #-- Si una f-string se come las llaves, aquí aparecerían los datos.

    def test_catalogo_regularizado_sin_rampas_repetidas(case):
        ajuste = rampas_regularizadas(*muestra())
        pares = set(zip(np.round(ajuste.inicios, 12), np.round(ajuste.anchos, 12)))
        case.assertEqual(len(pares), len(ajuste.inicios))


if __name__ == "__main__":
    unittest.main()
