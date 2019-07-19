from pprint import pprint

import mot
from mot.classes import State


def main():
    data = mot.parse("data/20190716T195132.csv")
    state = State.from_ob(data.pop(0))
    p_state = state.predict_at(data[0].time)
    pprint(p_state.distance(data[0]))
    pprint(p_state.distance(data[1]))
    pprint(p_state.distance(data[2]))


if __name__ == "__main__":
    main()
