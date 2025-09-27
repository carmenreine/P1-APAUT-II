"""
Planta de la práctica 1: mRMR (Minimum Redundancy Maximum Relevance)
Instrucciones:
- Completa la implementación de la clase mRMR.
- Asegúrate de que todos los tests en test_practica_1.py pasen correctamente. Para ello ejecuta `pytest` en la terminal.

Requisitos:
- Python 3.9+
- numpy, scipy (opcional), scikit-learn, pandas (opcional para reportes)

Autor:
"""
from sklearn.feature_selection import mutual_info_regression
import numpy as np


class mRMR:
    def __init__(self, n_features: int):
        """ Inicializa el selector mRMR.
        Parámetros:
        - n_features: Número de características a seleccionar.
        """
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        self.n_features = n_features
        self.selected_ = []  # índices de las características seleccionadas

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Ajusta el selector mRMR a los datos.
        Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas objetivo.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("X e y deben tener el mismo número de muestras")

        # --- 1. Relevancia de cada feature respecto a y
        relevance = mutual_info_regression(X, y)

        # --- 2. Redundancia entre pares de atributos (MI entre features)
        mi_matrix = np.array([
            mutual_info_regression(X, X[:, j])
            for j in range(n_features)
        ])
        redundancy_matrix = (mi_matrix + mi_matrix.T) / 2
        np.fill_diagonal(redundancy_matrix, 0)

        # --- 3. Selección iterativa
        self.selected_ = [int(np.argmax(relevance))]  # el más relevante primero
        while len(self.selected_) < self.n_features:
            scores = []
            for j in range(n_features):
                if j in self.selected_:
                    continue
                redundancia_media = np.mean([redundancy_matrix[j, k] for k in self.selected_])
                score = relevance[j] - redundancia_media
                scores.append((score, j))
            best = max(scores, key=lambda x: x[0])[1]
            self.selected_.append(best)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Disminuye la matriz de características a las características seleccionadas.
        Parámetros:
        - X: Matriz de características de entrada.
        Retorna:
        - X_reduced: Matriz de características reducida.
        """
        if not self.selected_:
            raise RuntimeError("Debes llamar a fit antes de transform")
        return np.array(X)[:, self.selected_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        """Devuelve los parámetros del modelo en un diccionario."""
        return {"n_features": self.n_features}

    def set_params(self, **params):
        """Permite modificar parámetros del modelo."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
