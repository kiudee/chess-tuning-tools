import numpy as np


def parse_timecontrol(tc_string):
    return tuple([float(x) for x in tc_string.split("+")])