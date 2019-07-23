import mot
from mot.classes import Tracker
from pprint import pprint


def main():
    observations = mot.parse("data/20190724T163125.csv")
    tracker = Tracker(observations)
    for i in range(5):
        tracker.step()
    pprint(tracker.hyps)


if __name__ == "__main__":
    main()
