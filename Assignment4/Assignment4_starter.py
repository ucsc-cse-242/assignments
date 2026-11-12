"""Optional starter code for CSE 242 Assignment 4.

You do not need to use this file. Add, remove, or reorganize functions as useful.
"""

import numpy as np


def update_assignments(X, centroids):
    """Return the index of the closest centroid for each observation."""
    raise NotImplementedError


def update_centroids(X, assignments, k):
    """Return updated centroids for k clusters."""
    raise NotImplementedError


def kmeans(X, k, max_iterations=100, tol=1e-6, random_state=None):
    """Fit K-means from scratch and return centroids, assignments, and history."""
    raise NotImplementedError


def corrupt_labels(y, noise_config, random_state=None):
    """Create a reproducible noisy version of y according to your design."""
    raise NotImplementedError


def repair_labels(X, y_noisy, **kwargs):
    """Optional structure for your nearest-neighbor or clustering repair method."""
    raise NotImplementedError
