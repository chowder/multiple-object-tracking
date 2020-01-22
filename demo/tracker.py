import matplotlib.pyplot as plt
import numpy as np
from heapq import nsmallest
from matplotlib.patches import Ellipse
from typing import List

from mot.classes import Hypothesis, Observation, Track

plt.ion()
PRUNE_SIZE = 10


class RealtimeTrackerDemo(object):
    def __init__(self):
        self.observations: List[Observation] = []
        self.hyps: List[Hypothesis] = []
        self.time = 0
        # Plotting stuff
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def add(self, observation: Observation):
        self.observations.append(observation)
        self.time = observation.time
        new_hyps = []
        for hyp in self.hyps:
            new_hyps = new_hyps + hyp.consider(observation)
        if len(new_hyps) == 0:
            hyp = Hypothesis()
            hyp.tracks.append(Track.from_observation(observation))
            new_hyps.append(hyp)
        self.hyps = new_hyps
        self.plot()
        self.prune()

    def prune(self):
        self.hyps = nsmallest(PRUNE_SIZE, self.hyps, key=lambda x: x.cost)

    def plot(self):
        self.ax.clear()
        hyp = self.hyps[0]
        for track in hyp.tracks:
            state, cov = track.predict_at(self.time)
            # Create Ellipse
            lambda_, v = np.linalg.eig(cov[0:2, 0:2])
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(state[0][0], state[1][0]),
                          width=lambda_[0]*5, height=lambda_[1]*5,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          edgecolor='b')
            ell.set_facecolor('none')
            self.ax.add_artist(ell)
            plt.scatter([state[0][0]], [state[1][0]], linewidths=1)
        plt.ylim(top=20, bottom=-20)
        plt.xlim(left=-20, right=20)
        plt.pause(0.0001)

