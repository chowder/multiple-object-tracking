import numpy as np


def Process(t: float):
    """
    Constant-velocity state process matrix
    """
    return np.array([
        [1, 0, t, 0],
        [0, 1, 0, t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])


"""
Measurement matrix, H
"""
Measurement = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]])

"""
Process noise matrix, Q
"""
ProcessNoise = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.1]])

"""
Initial covariance matrix, P0
"""
InitialCovariance = np.array([
    [5.0, 0.0, 0.0, 0.0],
    [0.0, 5.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.5]])
