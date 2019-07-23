import csv
from .classes import Observation


def _read(filename: str):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) # skip header
        data = [row for row in csvreader]
        return data


def parse(filename: str):
    data = _read(filename)
    obs = [Observation(float(i[0]), float(i[1]), float(i[2]), i[3], index) for index, i in enumerate(data)]
    return obs
