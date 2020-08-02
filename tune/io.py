from ast import literal_eval
from collections.abc import MutableMapping
import json
from pathlib import Path
import re
import subprocess
import sys

import skopt.space as skspace
from skopt.space.space import check_dimension


__all__ = [
    "InitStrings",
    "uci_tuple",
]


# TODO: Backup file to restore it, should there be an error
def uci_tuple(uci_string):
    try:
        name, value = re.findall(r"name\s+(\S.*?)\s+value\s+(.*?)\s*$", uci_string)[0]
    except IndexError:
        print(f"Error parsing UCI tuples:\n{uci_string}")
        sys.exit(1)
    try:
        tmp = float(value)
    except ValueError:
        tmp = value
    return name, tmp


def set_option(name, value):
    return f"setoption name {name} value {value}"


class InitStrings(MutableMapping):
    def __init__(self, init_strings):
        self._init_strings = init_strings

    def __len__(self):
        return len(self._init_strings)

    def __getitem__(self, key):
        for s in self._init_strings:
            name, value = uci_tuple(s)
            if key == name:
                return value
        raise KeyError(key)

    def __setitem__(self, key, value):
        for i, s in enumerate(self._init_strings):
            name, _ = uci_tuple(s)
            if key == name:
                self._init_strings[i] = set_option(key, value)
                return
        self._init_strings.append(set_option(key, value))

    def __delitem__(self, key):
        elem = -1
        for i, s in enumerate(self._init_strings):
            name, _ = uci_tuple(s)
            if key == name:
                elem = i
                break
        if elem != -1:
            del self._init_strings[i]
        else:
            raise KeyError(key)

    def __contains__(self, key):
        for s in self._init_strings:
            name, _ = uci_tuple(s)
            if key == name:
                return True
        return False

    def __iter__(self):
        for s in self._init_strings:
            name, _ = uci_tuple(s)
            yield name

    def __repr__(self):
        return repr(self._init_strings)


def make_numeric(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_ranges(s):
    if isinstance(s, str):
        j = json.loads(s)
    else:
        j = s

    dimensions = []
    for i, s in enumerate(j.values()):
        # First check, if the string is a list/tuple or a function call:
        param_str = re.findall(r"(\w+)\(", s)
        if len(param_str) > 0:  # Function
            args, kwargs = [], dict()
            # TODO: this split does not always work (example Categorical(["a", "b", "c"]))
            prior_param_strings = re.findall(r"\((.*?)\)", s)[0].split(",")
            for arg_string in prior_param_strings:
                # Check if arg or kwarg:
                if "=" in arg_string:  # kwarg:
                    # trim all remaining whitespace:
                    arg_string = "".join(arg_string.split())

                    key, val = arg_string.split("=")
                    kwargs[key] = make_numeric(val)
                elif "[" in arg_string or "(" in arg_string:
                    args.append(literal_eval(arg_string))
                else:  # args:
                    val = make_numeric(arg_string)
                    args.append(val)
            if hasattr(skspace, param_str[0]):
                dim = getattr(skspace, param_str[0])(*args, **kwargs)
            else:
                raise ValueError("Dimension {} does not exist.".format(param_str))
            dimensions.append(dim)
        else:  # Tuple or list
            # We assume that the contents of the collection should be used as is and construct a python list/tuple
            # skopt.space.check_dimension will be used for validation
            parsed = literal_eval(s)
            if isinstance(parsed, (tuple, list)):
                dimensions.append(check_dimension(parsed))
            else:
                raise ValueError(
                    "Dimension {} is not valid. Make sure you pass a Dimension, tuple or list.".format(
                        param_str
                    )
                )

    return dict(zip(j.keys(), dimensions))


def load_tuning_config(json_dict):
    """ Load the provided tuning configuration and split it up.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing the engines, their fixed parameters, the tunable
        parameter ranges and other settings used during tuning.

    Returns
    -------
    json_dict : dict
        Remaining settings after the engine configuration and the ranges have
        been stripped off
    commands : list of strings
    fixed_params : list of dicts
        UCI parameters to be set for the engines.
    param_ranges : dict
        UCI parameters of the first engine which are to be optimized during tuning.
        The values correspond to skopt.space dimension definitions.
    """
    commands = []
    fixed_params = []
    if "engines" not in json_dict:
        raise ValueError("Tuning config does not contain engines.")
    engines = json_dict["engines"]
    for e in engines:
        if "command" not in e:
            raise ValueError("Tuning config contains an engine without command.")
        commands.append(e["command"])
        if "fixed_parameters" not in e:
            fixed_params.append(dict())
        else:
            fixed_params.append(e["fixed_parameters"])
    del json_dict["engines"]
    if "parameter_ranges" not in json_dict:
        raise ValueError("There are no parameter ranges defined in the config file.")
    param_ranges = parse_ranges(json_dict["parameter_ranges"])
    del json_dict["parameter_ranges"]
    # All remaining settings will be returned as is:
    return json_dict, commands, fixed_params, param_ranges


def prepare_engines_json(commands, fixed_params):
    result_list = [
        {"command": c, "name": f"engine{i+1}", "initStrings": [], "protocol": "uci"}
        for i, c in enumerate(commands)
    ]
    for r, fp in zip(result_list, fixed_params):
        uci = InitStrings(r["initStrings"])
        uci.update(fp)
    return result_list


def write_engines_json(engine_json, point_dict):
    engine = engine_json[0]
    initstr = InitStrings(engine["initStrings"])
    initstr.update(point_dict)
    with open(Path() / "engines.json", "w") as file:
        json.dump(engine_json, file, sort_keys=True, indent=4)


class Engines(object):
    def __init__(self, parameters, engine1_fixed, engine2_fixed):
        pass

    def set_params(self, values):
        """Set tunable parameters to new values.

        Parameters
        ----------
        values :

        """
        pass

    def write_json(self, path=None):
        """Write the engines.json file

        Parameters
        ----------
        path :

        """
        pass
