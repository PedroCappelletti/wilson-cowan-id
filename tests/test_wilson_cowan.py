"""Tests del modelo de Wilson-Cowan."""

import pytest

from src.wilson_cowan import WilsonCowan, WilsonCowanParams


def test_default_params():
    wc = WilsonCowan()
    assert isinstance(wc.params, WilsonCowanParams)


@pytest.mark.skip(reason="Pendiente: implementar simulate()")
def test_simulate_shapes():
    wc = WilsonCowan()
    t, states = wc.simulate(E0=0.1, I0=0.1, t_span=(0.0, 1.0), dt=0.01)
    assert states.shape[1] == 2
    assert t.shape[0] == states.shape[0]
