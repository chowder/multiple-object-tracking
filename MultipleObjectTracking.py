import mot

from mot.classes import Tracker
from plotter import Plotter
from utils import save


def main():
    file = "data/20_Agents_3min.csv"
    observations = mot.parse(file)
    print("Loaded {} observations".format(len(observations)))

    tracker = Tracker(observations[:])
    tracker.process()

    # Obtain our best hypothesis
    tracks = [
        track.points for track in tracker.hyps[0].tracks
    ]
    # Save our results
    save(tracks, "results/20_Agents_3min.txt")

    # Initialise the plotter
    plot = Plotter()
    plot.use_observations(observations)
    plot.use_tracks(tracks)

    # Evaluate
    plot.evaluate_tracks()
    plot.plot_and_show()


if __name__ == "__main__":
    main()
