import pytest

from tune.io import combine_nested_parameters, load_tuning_config


def test_load_tuning_config():
    testdict = {
        "engines": [
            {"command": "lc0", "fixed_parameters": {"CPuctBase": 13232, "Threads": 2}},
            {"command": "sf", "fixed_parameters": {"Threads": 8}},
        ],
        "parameter_ranges": {"CPuct": "Real(0.0, 1.0)"},
        "gp_samples": 100,
    }
    json_dict, commands, fixed_params, param_ranges = load_tuning_config(testdict)
    assert len(json_dict) == 1
    assert "gp_samples" in json_dict
    assert len(commands) == 2
    assert len(fixed_params) == 2
    assert len(param_ranges) == 1


def test_combine_nested_parameters():
    # A dict without nested parameters should be unchanged:
    testdict = {
        "UCIParameter1": 42.0,
        "UCIParameter2": 0.0,
    }
    result = combine_nested_parameters(testdict)
    assert len(result) == 2
    assert "UCIParameter1" in result
    assert "UCIParameter2" in result
    assert result["UCIParameter1"] == 42.0
    assert result["UCIParameter2"] == 0.0

    # Test a correct specification of nested parameters:
    testdict = {
        "UCIParameter1": 42.0,
        "UCIParameter2=composite(sub-parameter1)": 0.0,
        "UCIParameter2=composite(sub-parameter2)": 1.0,
    }
    result = combine_nested_parameters(testdict)
    assert len(result) == 2
    assert "UCIParameter2" in result
    assert result["UCIParameter2"] == "composite(sub-parameter1=0.0,sub-parameter2=1.0)"

    # Test an incorrect specification of nested parameters:
    testdict = {
        "UCIParameter1": 42.0,
        "UCIParameter2=composite(sub-parameter1)": 0.0,
        "UCIParameter2=other(sub-parameter2)": 1.0,
    }
    with pytest.raises(ValueError):
        combine_nested_parameters(testdict)
