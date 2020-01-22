import matplotlib.pyplot as plt

from typing import List
from mot.classes import Observation


class Plotter:
    def __init__(self):
        self.observations = None
        self.true_tracks = dict()
        self.tracks = None
        self.plt = plt.figure()

    def use_observations(self, observations: List[Observation]):
        self.observations = observations
        self._get_true_tracks()

    def use_tracks(self, tracks: List[int]):
        self.tracks = tracks

    def _get_true_tracks(self):
        for observation in self.observations:
            self.true_tracks[observation.mac] = self.true_tracks.get(observation.mac, []) + [observation.index]

    def _plot_true_tracks(self):
        for track in self.true_tracks.values():
            x = [self.observations[point].pos[0] for point in track]
            y = [self.observations[point].pos[1] for point in track]
            plt.plot(x, y)
        plt.show()

    def _plot_predicted_tracks(self):
        for track in self.tracks:
            x = [self.observations[point].pos[0] for point in track]
            y = [self.observations[point].pos[1] for point in track]
            plt.plot(x, y)
        plt.show()

    def evaluate_tracks(self):
        incorrect_edges = 0
        # Create a dictionary of predicted edges
        predicted_edges = {}
        for track in self.tracks:
            for i in range(len(track)):
                predicted_edges[track[i]] = track[i-1] if i > 0 else None
        # Evaluate against the true tracks
        for track in self.true_tracks.values():
            for i in range(1, len(track)):
                if predicted_edges[track[i]] != track[i-1]:
                    incorrect_edges += 1
        print("Incorrect edges: {}".format(incorrect_edges))
        print("Total edges: {}".format(len(predicted_edges)))

    def plot_and_show(self):
        self._plot_true_tracks()
        self._plot_predicted_tracks()
