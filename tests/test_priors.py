import numpy as np
from pytest import approx, raises

from tune.priors import create_priors, make_invgamma_prior, roundflat


def test_roundflat():
    assert roundflat(0.3) == approx(0.0, abs=1e-6)
    assert roundflat(0.0) == -np.inf
    assert roundflat(-1.0) == -np.inf


def test_make_invgamma_prior():
    prior = make_invgamma_prior()
    assert prior.kwds["a"] == approx(8.919240823584246)
    assert prior.kwds["scale"] == approx(1.7290248731437994)

    with raises(ValueError):
        make_invgamma_prior(lower_bound=-1e-10)
    with raises(ValueError):
        make_invgamma_prior(upper_bound=-1e-10)
    with raises(ValueError):
        make_invgamma_prior(lower_bound=0.5, upper_bound=0.1)


def test_create_priors():
    priors = create_priors(n_parameters=3)
    assert len(priors) == 5
    assert priors[0](2.0) == approx(-1.536140897416146)
    assert priors[1](2.0) == approx(-23.620792572134874)
    assert priors[2](2.0) == approx(-23.620792572134874)
    assert priors[3](2.0) == approx(-23.620792572134874)
    assert priors[4](2.0) == approx(-10262570.41553909)

    with raises(ValueError):
        create_priors(n_parameters=3, signal_scale=0.0)
    with raises(ValueError):
        create_priors(n_parameters=3, noise_scale=0.0)
