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
from sklearn.feature_selection import mutual_info_classif
import numpy as np


class mRMR:
    def __init__(self, n_features: int):
        """
        Inicializa el selector mRMR.

<<<<<<< Updated upstream
        Parámetros
        ----------
        n_features : int
            Número de características a seleccionar.
        """
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        self.n_features = n_features
        self.selected_ = []
        self.selected_features_ = None
=======
        Parámetros:+
        - n_features: Número de características a seleccionar.
        """
        if n_features < 1:
            raise ValueError("n_features debe ser >= 1")
        self.n_features = n_features
        self.selected_idx = []
>>>>>>> Stashed changes

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el selector mRMR a los datos.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de características de entrada (n_samples x n_features).
        y : np.ndarray
            Vector de etiquetas objetivo (longitud n_samples).
        """
<<<<<<< Updated upstream
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if y.shape[0] != n_samples:
            raise ValueError("X e y deben tener el mismo número de muestras")

        # --- 1. Relevancia de cada feature respecto a y
        relevance = mutual_info_classif(X, y, discrete_features="auto", random_state=0)

        # --- 2. Redundancia entre pares de atributos
        mi_matrix = np.array([
            mutual_info_classif(
                X, (X[:, j] > np.median(X[:, j])).astype(int),
                discrete_features=True, random_state=0
            )
            for j in range(n_features)
        ])
        redundancy_matrix = (mi_matrix + mi_matrix.T) / 2
        np.fill_diagonal(redundancy_matrix, 0)

        # --- 3. Selección iterativa
        self.selected_ = [int(np.argmax(relevance))]
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

        self.selected_features_ = self.selected_
        return self

=======
        n_samples, n_features = X.shape
        relevance = mutual_info_regression(X, y, discrete_features="auto")

        selected = []
        remaining = list(range(n_features))

        # Seleccionamos primero la feature más relevante
        first = np.argmax(relevance)
        selected.append(first)
        remaining.remove(first)

        while len(selected) < self.n_features and remaining:
            scores = []
            for j in remaining:
                redundancy = np.mean([np.corrcoef(X[:, j], X[:, s])[0, 1]**2 
                                      for s in selected])
                redundancy = 0 if np.isnan(redundancy) else redundancy
                score = relevance[j] - redundancy
                scores.append((score, j))
            best = max(scores, key=lambda t: t[0])[1]
            selected.append(best)
            remaining.remove(best)

        self.selected_idx = selected
        return self
    
>>>>>>> Stashed changes
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve X reducido a las columnas seleccionadas.
        """
<<<<<<< Updated upstream
        if not self.selected_:
            raise RuntimeError("Debes llamar a fit antes de transform")
        return np.array(X)[:, self.selected_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo y devuelve la versión reducida de X.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        """Devuelve los parámetros del modelo (compatibilidad sklearn)."""
        return {"n_features": self.n_features}

    def set_params(self, **params):
        """Permite modificar parámetros del modelo."""
        for param, value in params.items():
            setattr(self, param, value)
=======
        if not self.selected_idx:
            raise RuntimeError("Debes llamar a fit antes de transform")
        return X[:, self.selected_idx]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


    def get_params(self, deep=True):
        return {"n_features": self.n_features}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
>>>>>>> Stashed changes
        return self
