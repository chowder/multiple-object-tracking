import numpy as np
import pprint

from copy import deepcopy
from heapq import nsmallest
from numpy.linalg import inv, multi_dot
from typing import List
from .models import Process, InitialCovariance, ProcessNoise, Measurement, MeasurementNoise

MAX_DISTANCE = 9.75
PRUNE_SIZE = 200
NEW_TRACK_COST = 100


class Observation(object):
    def __init__(self, time: float, x: float, y: float, mac: str, index: int):
        self.time = time
        self.pos = np.array([[x], [y]])
        self.state = np.array([[x], [y], [0], [0]])
        self.mac = mac
        self.index = index

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class Tracker(object):
    def __init__(self, observations: List[Observation]):
        self.observations = observations
        self.hyps: List[Hypothesis] = []

        # Initiate the first hypothesis (with 1 track of a single point)
        hyp = Hypothesis()
        hyp.tracks.append(Track.from_observation(observations.pop(0)))
        self.hyps.append(hyp)

    def step(self):
        observation = self.observations.pop(0)
        new_hyps = []
        for hyp in self.hyps:
            new_hyps = new_hyps + hyp.consider(observation)
        self.hyps = new_hyps
        self.prune()

    def prune(self):
        self.hyps = nsmallest(PRUNE_SIZE, self.hyps, key=lambda x: x.cost)

    def process(self):
        while len(self.observations) > 0:
            self.step()

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class Hypothesis(object):
    def __init__(self):
        self.tracks: List[Track] = []
        self.cost = 0

    def consider(self, observation: Observation):
        new_hyps = []
        for index, track in enumerate(self.tracks):
            compatible, new_track, cost = track.compatible_with(observation)
            if compatible:
                hyp = deepcopy(self)
                hyp.tracks[index] = new_track
                hyp.cost += cost
                new_hyps.append(hyp)

        # Consider that the observation may be a new track on its own
        hyp = deepcopy(self)
        hyp.tracks.append(Track.from_observation(observation))
        hyp.cost += 0 if len(new_hyps) == 0 else NEW_TRACK_COST
        new_hyps.append(hyp)
        return new_hyps

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class Track(object):
    def __init__(self, time: float, state: np.ndarray, cov: np.ndarray, index: int):
        self.points = [index]
        self.last_update_time = time
        self.state = state
        self.cov = cov

    @classmethod
    def from_observation(cls, o: Observation):
        """"
        Factory method to instantiate an initial state, x0, from an observation
        """
        return Track(o.time, o.state, InitialCovariance, o.index)

    def compatible_with(self, observation: Observation):
        # STEP 1: State prediction
        delta_time = observation.time - self.last_update_time
        f = Process(delta_time)
        pred_state = f.dot(self.state)
        pred_cov = multi_dot([f, self.cov, f.transpose()]) + ProcessNoise(delta_time)

        # STEP 2: Measurement prediction
        c = pred_cov[0:2, 0:2]
        v = Measurement.dot(observation.state - pred_state)
        distance = (multi_dot([v.transpose(), inv(c), v]) ** 0.5)[0][0]

        # STEP 3: Data Association
        if distance < MAX_DISTANCE:
            # print("Compatible! Distance: {:.3f}".format(distance))
            # STEP 4: State update
            new_track = deepcopy(self)
            new_track.points.append(observation.index)
            new_track.last_update_time = observation.time
            # Compute Kalman gain
            s = multi_dot([Measurement, pred_cov, Measurement.transpose()]) + MeasurementNoise
            k = multi_dot([pred_cov, Measurement.transpose(), inv(s)])
            # State and covariance update
            new_track.state = new_track.state + k.dot(v)
            new_track.cov = (np.identity(4) - k.dot(Measurement)).dot(new_track.cov)
            return True, new_track, distance
        else:
            # print("Not compatible! Distance: {:.3f}".format(distance))
            return False, None, 0

    def __repr__(self):
        return pprint.pformat(self.__dict__)
