"""Tests for which battery is filled first, and with how much.

The problem: charge power differs by an order of magnitude between an AC
battery and a hybrid inverter. Sharing a surplus by power limit hands the slow
one the smaller share — while it is the one needing the most hours to finish.
Observed on the reference installation: the 7 kW inverter full by early
afternoon, the 2.5 kW battery still at 52 % at sunset while a kilowatt went to
the grid.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.omnibattery import (
    _battery_remaining_kwh,
    _charge_order,
    _scarce_solar_day,
    _time_to_full_h,
)


class _Battery:
    """A stand-in coordinator. A class, not a namespace: the distributor keys
    its allocation dictionaries by coordinator."""

    def __init__(self, name, *, capacity, soc, limit_w, dc_coupled=False, max_soc=100):
        self.name = name
        self.battery_capacity_kwh = capacity
        self.max_soc = max_soc
        self.data = {"battery_soc": soc}
        self.is_available = True
        self.driver = SimpleNamespace(dc_coupled=dc_coupled)
        self._limit_w = limit_w


def _battery(name, **kwargs):
    return _Battery(name, **kwargs)


def _controller(batteries, *, forecast=None, avg_consumption=20.0, priority=""):
    return SimpleNamespace(
        coordinators=batteries,
        charge_priority=priority,
        _scarce_solar_latched=False,
        _active_charge_batteries=[],
        solar_forecast_sensor="sensor.forecast" if forecast is not None else None,
        _consumption_tracker=SimpleNamespace(
            get_avg_daily_consumption=lambda: avg_consumption
        ),
        hass=SimpleNamespace(states=SimpleNamespace(
            get=lambda _eid: SimpleNamespace(state=str(forecast))
            if forecast is not None else None
        )),
        _battery_power_limit=lambda coordinator, is_charging: coordinator._limit_w,
    )


# The reference installation, mid-afternoon: the hybrid full, the AC battery half.
def _reference():
    return [
        _battery("Marstek", capacity=15.36, soc=52, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=100, limit_w=7000, dc_coupled=True),
    ]


# ----------------------------------------------------------------------
# how long each battery still needs
# ----------------------------------------------------------------------
def test_remaining_energy_respects_the_configured_ceiling():
    battery = _battery("Marstek", capacity=15.36, soc=50, limit_w=2500, max_soc=90)
    assert _battery_remaining_kwh(battery) == pytest.approx(6.144)


def test_a_full_battery_wants_nothing():
    assert _battery_remaining_kwh(_reference()[1]) == 0.0


def test_hours_to_full_is_energy_over_power_not_state_of_charge():
    """The 7 kW inverter at 40 % is quicker than the 2.5 kW battery at 60 %."""
    controller = _controller([])
    slow = _battery("Marstek", capacity=15.36, soc=60, limit_w=2500)
    fast = _battery("Huawei", capacity=13.8, soc=40, limit_w=7000, dc_coupled=True)
    assert _time_to_full_h(controller, slow) == pytest.approx(2.458, abs=0.01)
    assert _time_to_full_h(controller, fast) == pytest.approx(1.183, abs=0.01)
    # Lower state of charge, yet done sooner — which is the whole point.
    assert _time_to_full_h(controller, fast) < _time_to_full_h(controller, slow)


def test_a_battery_that_cannot_charge_needs_no_hours():
    controller = _controller([])
    blocked = _battery("Marstek", capacity=15.36, soc=10, limit_w=0)
    assert _time_to_full_h(controller, blocked) == 0.0


# ----------------------------------------------------------------------
# the order
# ----------------------------------------------------------------------
def test_with_sun_to_spare_the_slow_battery_goes_first():
    batteries = [
        _battery("Marstek", capacity=15.36, soc=20, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=20, limit_w=7000, dc_coupled=True),
    ]
    controller = _controller(batteries, forecast=60.0, avg_consumption=20.0)
    assert [c.name for c in _charge_order(controller, batteries)] == ["Marstek", "Huawei"]


def test_on_a_thin_day_the_efficient_battery_goes_first():
    """Scarce kilowatt-hours belong where the least of them is lost."""
    batteries = [
        _battery("Marstek", capacity=15.36, soc=20, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=20, limit_w=7000, dc_coupled=True),
    ]
    controller = _controller(batteries, forecast=22.0, avg_consumption=20.0)
    assert [c.name for c in _charge_order(controller, batteries)] == ["Huawei", "Marstek"]


def test_a_nominated_battery_overrides_both_rules():
    batteries = _reference()
    controller = _controller(batteries, forecast=22.0, priority="Marstek")
    assert _charge_order(controller, batteries)[0].name == "Marstek"


def test_a_wandering_forecast_does_not_reshuffle_the_order():
    """A forecast moves all day; the verdict may not follow every step."""
    batteries = [
        _battery("Marstek", capacity=10.0, soc=0, limit_w=2500),
        _battery("Huawei", capacity=10.0, soc=0, limit_w=7000, dc_coupled=True),
    ]
    controller = _controller(batteries, forecast=41.0, avg_consumption=20.0)
    # Wants 20 kWh, expects 21: ample, but only just.
    assert _scarce_solar_day(controller) is False

    for forecast in (40.5, 39.5, 40.0, 39.0):
        controller.hass.states.get = (
            lambda _eid, value=forecast: SimpleNamespace(state=str(value))
        )
        assert _scarce_solar_day(controller) is False, forecast

    # Clearly short of the need, by more than the band: now it flips.
    controller.hass.states.get = lambda _eid: SimpleNamespace(state="30.0")
    assert _scarce_solar_day(controller) is True


def test_without_a_forecast_there_is_no_opinion():
    batteries = _reference()
    controller = _controller(batteries, forecast=None)
    assert _scarce_solar_day(controller) is False


# ----------------------------------------------------------------------
# how the surplus is split
# ----------------------------------------------------------------------
def _distributor(batteries):
    from custom_components.omnibattery.control.power_distribution import PowerDistribution

    controller = _controller(batteries, forecast=60.0)
    controller._clamp_to_system_capacity = lambda power, _b, _c: power
    selector = PowerDistribution.__new__(PowerDistribution)
    selector._controller = controller
    selector._is_battery_manual_owned = lambda _c: False
    return selector, controller


def _split(batteries, watts):
    selector, _ = _distributor(batteries)
    allocation = selector._distribute_power_by_limits(watts, batteries, is_charging=True)
    return {c.name: allocation[c] for c in batteries}


def test_the_split_follows_the_room_left_not_the_power_rating():
    """4 kW across a 2.5 kW battery with room and a 7 kW one nearly full."""
    batteries = [
        _battery("Marstek", capacity=15.36, soc=20, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=90, limit_w=7000, dc_coupled=True),
    ]
    split = _split(batteries, 4000)
    # By power rating the slow one would have had 1053 W of this.
    assert split["Marstek"] == 2500
    assert split["Huawei"] == 1500


def test_two_alike_batteries_still_share_evenly():
    batteries = [
        _battery("A", capacity=10.0, soc=50, limit_w=1000),
        _battery("B", capacity=10.0, soc=50, limit_w=1000),
    ]
    assert _split(batteries, 1000) == {"A": 500, "B": 500}


def test_batteries_are_aimed_at_finishing_together():
    """Shares proportional to the room left means one finish time, not two."""
    batteries = [
        _battery("Marstek", capacity=15.36, soc=67.4, limit_w=2500),   # 5.0 kWh left
        _battery("Huawei", capacity=13.8, soc=0, limit_w=7000, dc_coupled=True),
    ]
    split = _split(batteries, 4000)
    hours = {
        "Marstek": 5.0 / (split["Marstek"] / 1000),
        "Huawei": 13.8 / (split["Huawei"] / 1000),
    }
    assert hours["Marstek"] == pytest.approx(hours["Huawei"], rel=0.02)


def test_a_full_battery_is_passed_over():
    batteries = _reference()
    split = _split(batteries, 2000)
    assert split == {"Marstek": 2000, "Huawei": 0}


def test_nothing_is_left_on_the_table():
    """Surplus stops flowing only once every battery is at its limit."""
    batteries = [
        _battery("Marstek", capacity=15.36, soc=0, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=0, limit_w=7000, dc_coupled=True),
    ]
    assert sum(_split(batteries, 6000).values()) == 6000


def test_an_unknown_capacity_falls_back_to_the_old_share():
    batteries = [
        _battery("A", capacity=0, soc=50, limit_w=1000),
        _battery("B", capacity=0, soc=50, limit_w=3000),
    ]
    for battery in batteries:
        battery.data = {}
    assert _split(batteries, 2000) == {"A": 500, "B": 1500}


def test_discharge_still_shares_by_power():
    """Only charging changed; discharge is a question of who can deliver now."""
    selector, _ = _distributor([])
    batteries = [
        _battery("Marstek", capacity=15.36, soc=50, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=50, limit_w=7000, dc_coupled=True),
    ]
    allocation = selector._distribute_power_by_limits(4750, batteries, is_charging=False)
    assert {c.name: allocation[c] for c in batteries} == {"Marstek": 1250, "Huawei": 3500}


def test_both_new_selects_are_named_in_every_language():
    import glob
    import json

    for path in ["custom_components/omnibattery/strings.json"] + sorted(
        glob.glob("custom_components/omnibattery/translations/*.json")
    ):
        select = json.load(open(path, encoding="utf-8"))["entity"]["select"]
        assert select["charge_priority"]["name"], path
        assert select["charge_priority"]["state"]["automatic"], path


def test_the_panel_explains_the_new_control_in_every_language():
    import re

    panel = open(
        "custom_components/omnibattery/frontend/marstek-panel.js", encoding="utf-8"
    ).read()
    assert '{ key: "charge_priority", domain: "select"' in panel
    assert len(re.findall(r"\bitemChargePriority:", panel)) == 6
    assert len(re.findall(r'^    charge_priority: "', panel, re.M)) == 6
