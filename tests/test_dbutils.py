#!/usr/bin/env python

from decimal import Decimal

import numpy as np
from numpy.testing import assert_almost_equal

from tune.db_workers.utils import (
    penta,
    ldw_probabilities,
    draw_rate_to_elo,
    compute_probabilities_for_bias,
    compute_probabilities,
    elo_to_bayeselo,
    penta_to_score,
    TimeControl,
)


def test_penta():
    ldw1 = np.array([0.1, 0.2, 0.7])
    ldw2 = np.array([0.2, 0.2, 0.6])

    result = penta(ldw1, ldw2)
    expected = np.array([0.02, 0.06, 0.24, 0.26, 0.42])
    assert_almost_equal(result, expected, decimal=3)


def test_ldw_probabilities():
    result = ldw_probabilities(elo=50, draw_elo=200, bias=200)
    expected = np.array([0.06975828735890623, 0.35877859523271227, 0.5714631174083815])
    assert_almost_equal(result, expected)


def test_draw_rate_to_elo():
    result = draw_rate_to_elo(0.5)
    expected = np.array(190.84850188786498)
    assert_almost_equal(result, expected)


def test_compute_probabilities_for_bias():
    result = compute_probabilities_for_bias(elo=50, draw_elo=200, bias=200)
    expected = np.array([0.029894, 0.18540627, 0.41591514, 0.30154527, 0.06723932])
    assert_almost_equal(result, expected)


def test_compute_probabilities():
    result = compute_probabilities(elo=50, draw_elo=200, biases=(0, 200))
    expected = np.array(
        [
            0.033318056828286285,
            0.19078749500997028,
            0.3957332350793835,
            0.30255132321508954,
            0.07760988986727044,
        ]
    )
    assert_almost_equal(result, expected)


def test_elo_to_bayeselo():
    result = elo_to_bayeselo(elo=50, draw_elo=200, biases=(0, 200))
    expected = 71.513929
    assert_almost_equal(result, expected, decimal=5)


def test_penta_to_score():
    counts = np.array([1, 2, 3, 4, 5])
    result = penta_to_score(draw_rate=0.5, counts=counts, prior_games=10, prior_elo=0)
    expected = 0.4016368226279837
    assert_almost_equal(result, expected)


def test_timecontrol():
    strings = ("3.0+0.03", "7.0+0.0")
    result = TimeControl.from_strings(*strings)
    expected = (Decimal("3.0"), Decimal("0.03"), Decimal(7), Decimal(0))
    assert result == expected

    assert result.to_strings() == strings
