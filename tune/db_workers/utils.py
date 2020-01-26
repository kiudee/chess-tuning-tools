from collections import namedtuple
import numpy as np

MatchResult = namedtuple("MatchResult", ["wins", "losses", "draws"])
TimeControl = namedtuple("TimeControl", ["engine1", "engine2"])


def parse_timecontrol(tc_string):
    return tuple([float(x) for x in tc_string.split("+")])