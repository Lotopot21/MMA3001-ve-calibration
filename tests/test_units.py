"""Tests for unit conversion helpers."""

import pytest

from vetuner.units import afr_to_lambda


def test_stoichiometric_afr_gives_lambda_one():
    assert afr_to_lambda(14.7) == pytest.approx(1.0)


def test_rich_mixture_gives_lambda_below_one():
    assert afr_to_lambda(12.5) < 1.0


def test_negative_afr_raises():
    with pytest.raises(ValueError):
        afr_to_lambda(-1.0)