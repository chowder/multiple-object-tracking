from typing import List

import mot
from mot import Observation


def obs_to_tracklets(observations: List[Observation], size):
    tracks = []
    temp = {}
    for obs in observations:
        if len(temp.get(obs.mac, [])) >= size:
            tracks.append(temp.get(obs.mac))
            temp[obs.mac] = []
        temp[obs.mac] = temp.get(obs.mac, []) + [obs.index]
    return tracks


def load_predicted_tracks(filename: str):
    with open(filename) as f:
        lines = f.readlines()
        return [eval(line) for line in lines]


def tracklet_held(tracklet, pred_tracks):
    longest = 0
    current = 0
    last_track = find_point(tracklet[0], pred_tracks)
    for point in tracklet:
        track = find_point(point, pred_tracks)
        if last_track != track:
            last_track = track
            longest = max(current, longest)
            current = 0
        else:
            current += 1
    return longest / len(tracklet)


def find_point(p, pred_tracks):
    for idx, track in enumerate(pred_tracks):
        for point in track:
            if p == point:
                return idx


def track_hold(tracklets: List, pred_tracks: List, upper: float = 0.8, lower: float = 0.2):
    res = [tracklet_held(tracklet, pred_tracks) for tracklet in tracklets]
    mostly_tracked = sum(i >= upper for i in res)
    mostly_lost = sum(i <= lower for i in res)
    partially_tracked = len(res) - mostly_tracked - mostly_lost
    return sum(res) / len(res), mostly_tracked, partially_tracked, mostly_lost


def analyse(dataFile: str, resultFile: str, size=30):
    obsservations = mot.parse(dataFile)
    tracklets = obs_to_tracklets(obsservations, size)
    predicted_tracks = load_predicted_tracks(resultFile)
    res = track_hold(tracklets, predicted_tracks)
    print(res)


# analyse("../data/20_Agents_3min.csv", "../results/20_Agents_3min.txt", 30)
