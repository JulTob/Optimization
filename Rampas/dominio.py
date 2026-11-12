"""Los modelos 2D viven en el cuadrante positivo: el mínimo de cada variable es el origen."""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Escala:
    minimo: np.ndarray
    rango: np.ndarray

    @staticmethod
    def de_los_datos(X):
        """Se calcula SOLO con los datos de entrenamiento."""
        minimo = X.min(axis=0)
        rango = np.ptp(X, axis=0)
        if np.any(rango == 0):
            raise ValueError("Una variable es constante: no hay nada que ordenar.")
        return Escala(minimo, rango)

    def __call__(self, X):
        return (X - self.minimo) / self.rango
