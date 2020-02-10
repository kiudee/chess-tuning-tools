import re
import sys
from collections.abc import MutableMapping


# TODO: Backup file to restore it, should there be an error
def uci_tuple(uci_string):
    try:
        name, value = re.findall(r"name (\w+) value (\S*)", uci_string)[0]
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
