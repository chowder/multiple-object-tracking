import numpy as np
import pprint
from numpy.linalg import inv, multi_dot

from .models import Process, InitialCovariance, ProcessNoise, Measurement


class Observation(object):
    def __init__(self, time: float, x: float, y: float, mac: str):
        self.time = time
        self.pos = np.array([[x], [y]])
        self.state = np.array([[x], [y], [0], [0]])
        self.mac = mac

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class State(object):
    def __init__(self, time: float, mac: str, state: np.ndarray, cov: np.ndarray = None):
        self.time = time
        self.mac = mac
        self.state = state
        self.cov = InitialCovariance if cov is None else cov

    @classmethod
    def from_ob(cls, o: Observation):
        """"
        Factory method to instantiate an initial state, x0, from an observation
        """
        return State(o.time, o.mac, o.state)

    def predict_at(self, time: float):
        """
        Returns the predicted state at time `time`
        """
        delta_time = time - self.time
        f = Process(delta_time)
        # Perform predictions
        pred_state = f.dot(self.state)
        pred_cov = multi_dot([f, self.cov, f.transpose()]) + ProcessNoise(delta_time)
        return State(time, self.mac, pred_state, pred_cov)

    def distance(self, other: Observation) -> float:
        c = self.cov[0:2, 0:2]
        v = Measurement.dot(self.state - other.state)
        r = multi_dot([v.transpose(), inv(c), v]) ** 0.5
        return r[0][0]

    def __repr__(self):
        return pprint.pformat(self.__dict__)
