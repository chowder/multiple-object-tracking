import numpy as np

MAX_DISTANCE = 10
PRUNE_SIZE = 400
NEW_TRACK_COST = 100
KEEP_ALIVE_FRAMES = 3


def Process(t: float):
    """
    Constant-velocity state process matrix
    """
    return np.array([
        [1, 0, t, 0],
        [0, 1, 0, t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])


def ProcessNoise(t: float):
    """
    Process noise matrix, Q
    """
    return np.array([
        [0.5 * t, 0.0, 0.0, 0.0],
        [0.0, 0.5 * t, 0.0, 0.0],
        [0.0, 0.0, 0.1 * t, 0.0],
        [0.0, 0.0, 0.0, 0.1 * t]])


"""
Measurement matrix, H
"""
Measurement = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]])


MeasurementNoise = np.array([
    [0.1, 0],
    [0, 0.1]
])


"""
Initial covariance matrix, P0
"""
InitialCovariance = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.1]])
