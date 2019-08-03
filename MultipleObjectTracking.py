import mot

from pprint import pprint
from mot.classes import Tracker
from plotter import Plotter


def main():
    file = "data/20190725T130638.csv"
    observations = mot.parse(file)

    tracker = Tracker(observations[:])
    tracker.process()

    tracks = [
        track.points for track in tracker.hyps[0].tracks
    ]

    plot = Plotter()
    plot.use_observations(observations)
    plot.use_tracks(tracks)
    # plot._evaluate_tracks()

    plot.plot_and_show()


if __name__ == "__main__":
    main()
