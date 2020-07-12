from numpy.testing import assert_almost_equal
from tune.io import load_tuning_config


def test_load_tuning_config():
    testdict = {
        "engines": [
            {
                "command": "lc0",
                "fixed_parameters": {
                    "CPuctBase": 13232,
                    "Threads": 2
                }
            },
            {
                "command": "sf",
                "fixed_parameters": {
                    "Threads": 8
                }
            }
        ],
        "parameter_ranges": {
            "CPuct": "Real(0.0, 1.0)"
        },
        "gp_samples": 100,
    }
    json_dict, commands, fixed_params, param_ranges = load_tuning_config(testdict)
    assert len(json_dict) == 1
    assert "gp_samples" in json_dict
    assert len(commands) == 2
    assert len(fixed_params) == 2
    assert len(param_ranges) == 1
