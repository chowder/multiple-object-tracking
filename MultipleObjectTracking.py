from pprint import pprint

import mot
from mot.classes import State, Hypothesis, Target


def main():
    observations = mot.parse("data/20190716T195132.csv")
    o = observations.pop(0)
    state = State.from_observation(o)

    initial_target = Target.from_initial_state(state)
    hyps = [Hypothesis()]
    hyps[0].add(initial_target)


if __name__ == "__main__":
    main()
