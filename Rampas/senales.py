"""La pieza común a todo el trabajo: la rampa saturada.

Notación de señales y sistemas: u(t) es el escalón unitario.
    r(t)       rampa unitaria saturada: sube de 0 a 1 entre t = 0 y t = 1.
    R_{a,b}(x) rampa general: sube de 0 a 1 entre x = a y x = b.
(Ojo: en señales, r(t) suele ser la rampa sin techo t·u(t); la nuestra satura en 1.)
"""
import numpy as np


def rampa_unitaria(t):
    """r(t) = 0 si t < 0;  t si 0 ≤ t ≤ 1;  1 si t > 1."""
    return np.clip(t, 0.0, 1.0)


def rampa(x, inicio, fin):
    """R_{a,b}(x) = r((x − a)/(b − a)). Sube de 0 a 1 entre a y b."""
    return rampa_unitaria((x - inicio) / (fin - inicio))


def matriz_de_rampas(x, inicios, anchos):
    """Columna j = rampa que empieza en inicios[j] y dura anchos[j]."""
    x = np.asarray(x, dtype=float)
    return rampa_unitaria((x[:, None] - inicios) / anchos)
