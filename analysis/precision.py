from typing import List

import numpy as np

import mot

from numpy.linalg import multi_dot, inv

from analysis.completeness import load_predicted_tracks
from mot import models
from mot.classes import Observation


class KalmanFilter:
    def __init__(self, observation: Observation):
        self.state = observation.state
        self.time = observation.time
        self.cov = models.InitialCovariance

    def distance(self, observation: Observation) -> float:
        # Innovation
        v = models.Measurement.dot(observation.state - self.state)
        # The Euclidean distance
        distance = ((v[0]**2 + v[1]**2)**0.5)[0]
        return distance

    def predict_at(self, time: float):
        delta_time = time - self.time
        f = models.Process(delta_time)
        pred_state = f.dot(self.state)
        pred_cov = multi_dot([f, self.cov, f.transpose()]) + models.ProcessNoise(delta_time)
        return pred_state, pred_cov

    def update(self, observation: Observation):
        # Predict
        pred_state, pred_cov = self.predict_at(observation.time)
        # Kalman Gain
        s = multi_dot([models.Measurement, pred_cov, models.Measurement.transpose()]) + models.MeasurementNoise
        k = multi_dot([pred_cov, models.Measurement.transpose(), inv(s)])
        v = models.Measurement.dot(observation.state - pred_state)
        # State update
        self.state = self.state + k.dot(v)
        self.cov = (np.identity(4) - k.dot(models.Measurement)).dot(self.cov)
        self.time = observation.time


def load_actual_tracks(observations: List[Observation]):
    tracks = {}
    for obs in observations:
        tracks[obs.mac] = tracks.get(obs.mac, []) + [obs.index]
    return [track for track in tracks.values()]


def tracking_distance_error(track, observations):
    # Initialise KF
    kf = KalmanFilter(observations[track[0]])
    distance = kf.distance(observations[track[0]])
    for point in track[1:]:
        kf.update(observations[point])
        distance += kf.distance(observations[point])
    return distance


def analyse(data_file: str, results_file: str):
    observations = mot.parse(data_file)
    tracks = load_predicted_tracks(results_file)
    # tracks = load_actual_tracks(observations)
    total = 0
    for track in tracks:
        total += tracking_distance_error(track, observations)
    return total / len(observations)


print(analyse("../data/5_Agents_5min.csv", "../results/5_Agents_5min.txt"))
print(analyse("../data/10_Agents_10min.csv", "../results/10_Agents_10min.txt"))
print(analyse("../data/15_Agents_4min.csv", "../results/15_Agents_4min.txt"))
print(analyse("../data/20_Agents_3min.csv", "../results/20_Agents_3min.txt"))
