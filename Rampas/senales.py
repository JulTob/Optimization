"""La pieza común a todo el trabajo: la rampa saturada."""
import numpy as np


def rho(t):
    """ρ(t) = 0 si t < 0;  t si 0 ≤ t ≤ 1;  1 si t > 1."""
    return np.clip(t, 0.0, 1.0)


def rampa(x, inicio, fin):
    """R_{a,b}(x) = ρ((x − a)/(b − a)). Sube de 0 a 1 entre a y b."""
    return rho((x - inicio) / (fin - inicio))


def matriz_de_rampas(x, inicios, anchos):
    """Columna j = rampa que empieza en inicios[j] y dura anchos[j]."""
    x = np.asarray(x, dtype=float)
    return rho((x[:, None] - inicios) / anchos)
