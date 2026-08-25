"""A device that reports a power ceiling of zero has not answered.

A Marstek Venus came back from a restart with registers 44002 and 44003 both
reading 0. The coordinator adopted them as the configured ceiling, which shut
the battery out of every allocation — it could neither charge nor discharge, and
nothing would ever write those registers again, because the controller had
stopped addressing a battery it believed could do nothing.
"""
from __future__ import annotations

import pytest

from custom_components.omnibattery.infra.coordinator import (
    MarstekVenusDataUpdateCoordinator,
    _sync_device_reported_limits,
)


class _Coordinator:
    """The narrow slice of the coordinator this sync touches."""

    def __init__(self, data):
        self.name = "Omnibattery Marstek Venus"
        self.data = data
        self.needs_software_max_charge = False
        self.needs_software_max_discharge = False
        self.needs_software_power_cap = False
        self._configured_max_charge_power = 2500
        self._configured_max_discharge_power = 2500
        self._device_max_charge_power = 2500
        self._device_max_discharge_power = 2500
        self._effective_max_charge_power = 2500
        self._effective_max_discharge_power = 2500
        self.min_soc = 10
        self.max_soc = 100

    configured_max_charge_power = MarstekVenusDataUpdateCoordinator.configured_max_charge_power
    configured_max_discharge_power = MarstekVenusDataUpdateCoordinator.configured_max_discharge_power
    device_max_charge_power = MarstekVenusDataUpdateCoordinator.device_max_charge_power
    device_max_discharge_power = MarstekVenusDataUpdateCoordinator.device_max_discharge_power
    _recompute_effective_power_limits = (
        MarstekVenusDataUpdateCoordinator._recompute_effective_power_limits
    )


def _sync(data):
    coordinator = _Coordinator(data)
    _sync_device_reported_limits(coordinator)
    return coordinator


def test_a_reported_ceiling_of_zero_is_not_adopted():
    """The exact state after the restart: both registers reading 0."""
    coordinator = _sync({"max_charge_power": 0, "max_discharge_power": 0})
    assert coordinator._effective_max_charge_power == 2500
    assert coordinator._effective_max_discharge_power == 2500


def test_a_real_ceiling_is_still_adopted():
    coordinator = _sync({"max_charge_power": 1500, "max_discharge_power": 800})
    assert coordinator._effective_max_charge_power == 1500
    assert coordinator._effective_max_discharge_power == 800


def test_one_register_answering_does_not_drag_the_other_down():
    coordinator = _sync({"max_charge_power": 0, "max_discharge_power": 1200})
    assert coordinator._effective_max_charge_power == 2500
    assert coordinator._effective_max_discharge_power == 1200


def test_the_mismatch_is_logged_rather_than_swallowed(caplog):
    """Silence here cost a day of a battery sitting idle in the sun."""
    import logging

    with caplog.at_level(logging.WARNING):
        _sync({"max_charge_power": 0})
    assert "max_charge_power = 0" in caplog.text
    assert "restarts" in caplog.text or "restart" in caplog.text
