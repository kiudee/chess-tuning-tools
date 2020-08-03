from collections import namedtuple
from contextlib import contextmanager
from decimal import Decimal

import numpy as np
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None
from scipy.optimize import root_scalar
from scipy.special import expit

from tune.utils import parse_timecontrol

MatchResult = namedtuple("MatchResult", ["wins", "losses", "draws"])

ELO_CONSTANT = np.log(10) / 400.0


TC = namedtuple(
    "TimeControl",
    ["engine1_time", "engine1_increment", "engine2_time", "engine2_increment"],
)


class TimeControl(TC):
    @classmethod
    def from_strings(cls, engine1, engine2):
        tc1 = parse_timecontrol(engine1)
        tc2 = parse_timecontrol(engine2)
        inc1 = Decimal("0.0") if len(tc1) == 1 else tc1[1]
        inc2 = Decimal("0.0") if len(tc2) == 1 else tc2[1]
        return cls(
            engine1_time=Decimal(tc1[0]),
            engine1_increment=inc1,
            engine2_time=Decimal(tc2[0]),
            engine2_increment=inc2,
        )

    def to_strings(self):
        return (
            f"{self.engine1_time}+{self.engine1_increment}",
            f"{self.engine2_time}+{self.engine2_increment}",
        )


def get_session_maker(sessionmaker):
    @contextmanager
    def make_session():
        session = sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return make_session


def create_sqlalchemy_engine(config):
    db_uri = (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['dbname']}"
    )
    return create_engine(db_uri, pool_pre_ping=True)


def penta(ldw1, ldw2):
    x = np.fliplr(ldw1[:, None] * ldw2[None, :])
    sum_diagonals = np.vectorize(lambda offset: np.trace(x, offset=offset))
    ind = np.arange(-2, 3)[::-1]
    return sum_diagonals(ind)


def score_in_01(rates):
    k = len(rates)
    return np.sum(np.arange(k) * rates) / (k - 1)


def ldw_probabilities(elo, draw_elo, bias):
    pos = elo + bias
    w = expit(ELO_CONSTANT * (pos - draw_elo))
    l = expit(ELO_CONSTANT * (-pos - draw_elo))
    d = 1 - w - l
    return np.array([l, d, w])


def draw_rate_to_elo(draw_rate):
    return 400 * np.log10(2 / (1 - draw_rate) - 1)


def compute_probabilities_for_bias(elo, draw_elo, bias):
    ldw1 = ldw_probabilities(elo, draw_elo, bias)
    ldw2 = ldw_probabilities(elo, draw_elo, -bias)
    p = penta(ldw1, ldw2)
    return p


def compute_probabilities(elo, draw_elo, biases):
    result = np.empty((len(biases), 5))
    for i, b in enumerate(biases):
        result[i] = compute_probabilities_for_bias(elo, draw_elo, b)
    return result.mean(axis=0)


def elo_to_bayeselo(elo, draw_elo, biases):
    def func(bayeselo):
        return score_in_01(compute_probabilities(bayeselo, draw_elo, biases)) - expit(
            ELO_CONSTANT * elo
        )

    return root_scalar(func, method="brentq", bracket=(-1000, 1000)).root


def prior_from_drawrate(elo, draw_rate):
    biases = [-90, 200]
    draw_elo = draw_rate_to_elo(draw_rate)
    bayeselo = elo_to_bayeselo(elo, draw_elo=draw_elo, biases=biases)
    return compute_probabilities(bayeselo, draw_elo, biases)


def penta_to_score(draw_rate, counts, prior_games=10, prior_elo=0):
    prior = prior_games * prior_from_drawrate(prior_elo, draw_rate)
    probabilities = (counts + prior) / (counts.sum() + prior.sum())
    s01 = score_in_01(probabilities)
    score = (
        np.sqrt(2)
        * (s01 - 0.5)
        / (probabilities.dot(np.power(np.linspace(0, 2, 5), 2)) - 4 * s01 ** 2)
    )
    return score


def simple_penta_to_score(draw_rate, counts, prior_games=10, prior_elo=0):
    prior = prior_games * prior_from_drawrate(prior_elo, draw_rate)
    probabilities = (counts + prior) / (counts.sum() + prior.sum())
    s01 = score_in_01(probabilities)
    return s01 * 2 - 1
