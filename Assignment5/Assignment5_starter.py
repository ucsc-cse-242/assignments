"""Optional starter code for CSE 242 Assignment 5.

This is deliberately minimal. You may redesign the code structure completely.
"""

import numpy as np


class GaussianMixtureEM:
    def __init__(self, n_components, max_iter=100, tol=1e-5, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize(self, X):
        raise NotImplementedError

    def e_step(self, X):
        """Return posterior responsibilities for every example/component."""
        raise NotImplementedError

    def m_step(self, X, responsibilities):
        raise NotImplementedError

    def fit(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, random_state=None):
        raise NotImplementedError

    def forward(self, X):
        """Compute intermediate activations and predictions."""
        raise NotImplementedError

    def loss(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, X, y_true, cache):
        """Compute gradients by backpropagation."""
        raise NotImplementedError

    def step(self, gradients, learning_rate):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, learning_rate=0.01):
        raise NotImplementedError
