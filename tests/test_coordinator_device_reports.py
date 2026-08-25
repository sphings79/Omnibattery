"""What the coordinator makes of the figures a device reports.

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


# ----------------------------------------------------------------------
# daily totals across mixed fleets
#
# The system totals refuse to add up unless every battery's daily figure is
# marked as belonging to today — a guard against summing one battery's fresh
# value with another's from before midnight. Only the derived counter stamps
# that mark, because until the Huawei driver arrived every brand needed one.
#
# A battery whose device counts for itself never got that sensor, so it never
# carried the mark, and one of them in a fleet zeroed the totals outright: 10.3
# kWh charged on one battery and 3.97 on the other, the overview reading 0.00.
# ----------------------------------------------------------------------
from types import SimpleNamespace

from custom_components.omnibattery.infra.coordinator import (
    _stamp_native_daily_reset_dates,
)


def _native(data, has_daily=True):
    coordinator = SimpleNamespace(
        data=data, capabilities=SimpleNamespace(has_daily_energy_counters=has_daily)
    )
    _stamp_native_daily_reset_dates(coordinator)
    return coordinator.data


def test_a_device_counted_daily_value_is_marked_as_todays():
    from homeassistant.util import dt as dt_util

    today = dt_util.now().date().isoformat()
    data = _native({"total_daily_charging_energy": 10.3,
                    "total_daily_discharging_energy": 4.42})
    assert data["total_daily_charging_energy_reset_date"] == today
    assert data["total_daily_discharging_energy_reset_date"] == today


def test_a_driver_with_a_derived_counter_is_left_alone():
    """It stamps its own mark, and overwriting it would defeat the guard."""
    data = _native({"total_daily_charging_energy": 3.97}, has_daily=False)
    assert "total_daily_charging_energy_reset_date" not in data


def test_a_value_that_did_not_arrive_gets_no_mark():
    data = _native({"total_daily_discharging_energy": 4.42})
    assert "total_daily_charging_energy_reset_date" not in data
    assert "total_daily_discharging_energy_reset_date" in data


def test_zero_is_a_value_and_gets_marked():
    """Before the first charge of the day the figure is legitimately zero."""
    data = _native({"total_daily_charging_energy": 0.0})
    assert "total_daily_charging_energy_reset_date" in data
