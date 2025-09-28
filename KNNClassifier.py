#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plantilla de práctica: KNN, optimización de hiperparámetros y selección de atributos (mRMR)

Instrucciones:
- Completa la implementación de la clase KNNClassifier.
- Asegúrate de que todos los tests en test_practica_1.py pasen correctamente. Para ello ejecuta `pytest` en la terminal.

Requisitos:
- Python 3.9+
- numpy, scipy (opcional), scikit-learn, pandas (opcional para reportes)

Autor: Leire Bernárdez Vázquez, Carmen Reiné Rueda
"""
import numpy as np

class KNNClassifier:
    def __init__(self, k=3, distance_metric=None):
        """
        Inicializa el clasificador KNN.

        Parámetros:
        - k: Número de vecinos a considerar.
        - distance_metric: Función de distancia que toma dos vectores y devuelve un escalar.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un entero positivo")
        self.k = k
        self.distance_metric = distance_metric or self.euclidean_distance
        self.X_train = None
        self.y_train = None
        self._fitted = False

    @staticmethod
    def euclidean_distance(a, b):
        """Distancia euclídea entre dos arrays."""
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self, X, y):
        """
        Ajusta el modelo KNN a los datos de entrenamiento.

        Parámetros:
        - X: Matriz de características de entrenamiento.
        - y: Vector de etiquetas de entrenamiento.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X e y deben tener el mismo número de ejemplos")
        self._fitted = True
        
        return self


    def predict(self, X):
        """
        Predice las etiquetas para los patrones de X.

        Parámetros:
        - X: Lista de listas (M x d) de características.

        Devuelve:
        - Lista de etiquetas predichas (M).
        """
        
        if not self._fitted:
            raise RuntimeError("Debes llamar a fit antes de predecir")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for x in X:
            
            # Calcular todas las distancias
            dists = np.array([self.distance_metric(x, xi) for xi in self.X_train])
            
            # Vecinos más cercanos
            idx = np.argsort(dists)[:self.k]
            vecinos = self.y_train[idx]
            
            # Votación por mayoría
            values, counts = np.unique(vecinos, return_counts=True)
            pred = values[np.argmax(counts)]
            preds.append(pred)
            
        return np.array(preds)
    
    def get_params(self, deep=True):
        return {"k": self.k, "distance_metric": self.distance_metric}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def score(self, X, y):
        """
        Devuelve la accuracy del clasificador en los datos dados.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
