"""Wraps gradcheck.py's numerical-vs-analytical backprop check as a pytest
test -- the hand-rolled dqn.py backprop is checked against central
differences on 40 sampled parameters across all four weight tensors."""
from gradcheck import run


def test_backprop_matches_numerical_gradient():
    run()
