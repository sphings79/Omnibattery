"""Tests for which battery is filled first, and with how much.

The problem: charge power differs by an order of magnitude between an AC
battery and a hybrid inverter. Sharing a surplus by power limit hands the slow
one the smaller share — while it is the one needing the most hours to finish.
Observed on the reference installation: the 7 kW inverter full by early
afternoon, the 2.5 kW battery still at 52 % at sunset while a kilowatt went to
the grid.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


def _controller(
    batteries, *, forecast=None, avg_consumption=20.0, priority="",
    unit="kWh", produced_kwh=0.0, t_end=20.0, window_per_day=24.0,
    hours_ahead=None,
):
    """A controller stub carrying the pieces the outlook reads.

    ``hours_ahead`` is what the tracker reports as consumption-window hours
    still to come; left None it is derived from the clock, which is what makes
    the horizon tests sensitive to the time of day.
    """
    def _hours_in_range(now_h, end_h):
        if hours_ahead is not None:
            return hours_ahead
        return max(0.0, end_h - now_h)

    return SimpleNamespace(
        coordinators=batteries,
        charge_priority=priority,
        _scarce_solar_latched=False,
        _active_charge_batteries=[],
        _daily_solar_energy_kwh=produced_kwh,
        _solar_t_start=8.0,
        solar_forecast_sensor="sensor.forecast" if forecast is not None else None,
        _consumption_tracker=SimpleNamespace(
            get_avg_daily_consumption=lambda: avg_consumption,
            estimate_t_end=lambda: t_end,
            get_solar_fraction_done=lambda now_h, t_start, end_h: (
                0.0 if end_h <= (t_start or 0.0)
                else max(0.0, min(1.0, (now_h - (t_start or 0.0)) / (end_h - (t_start or 0.0))))
            ),
            get_consumption_window_hours_per_day=lambda: window_per_day,
            consumption_window_hours_in_range=_hours_in_range,
        ),
        hass=SimpleNamespace(states=SimpleNamespace(
            get=lambda _eid: SimpleNamespace(
                state=str(forecast), attributes={"unit_of_measurement": unit}
            )
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
    # Pinned rather than clock-derived: 60 kWh forecast with none produced yet
    # and the whole consumption window still ahead.
    controller = _controller(
        batteries, forecast=60.0, avg_consumption=20.0,
        produced_kwh=0.01, hours_ahead=24.0,
    )
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
    controller = _controller(
        batteries, forecast=41.0, avg_consumption=20.0,
        produced_kwh=0.01, hours_ahead=24.0,
    )

    def _forecast(value):
        return lambda _eid: SimpleNamespace(
            state=str(value), attributes={"unit_of_measurement": "kWh"}
        )

    # Wants 20 kWh, expects 21: ample, but only just.
    assert _scarce_solar_day(controller) is False

    for forecast in (40.5, 39.5, 40.0, 39.0):
        controller.hass.states.get = _forecast(forecast)
        assert _scarce_solar_day(controller) is False, forecast

    # Clearly short of the need, by more than the band: now it flips.
    controller.hass.states.get = _forecast(30.0)
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


def test_the_head_battery_reaches_its_own_rating():
    """What the surplus feedforward is for: the slow battery at full power.

    The reference case, mid-morning: the hybrid at 97 % with almost no room
    left, the AC battery at 21 % with 12 kWh of it. A system figure capped at
    2500 W left the AC battery on 2418 — its share of its own rating — while
    6.8 kW of surplus went past. Offering the fleet's whole capacity lets the
    distribution cap it at 2500 where it belongs.
    """
    batteries = [
        _battery("Marstek", capacity=15.36, soc=21, limit_w=2500),
        _battery("Huawei", capacity=13.8, soc=97, limit_w=7000, dc_coupled=True),
    ]
    assert _split(batteries, 2500)["Marstek"] == 2420    # the old, capped figure
    assert _split(batteries, 6828)["Marstek"] == 2500    # the whole surplus


# ----------------------------------------------------------------------
# the forecast: unit, and which stretch of day it describes
#
# The setup validates a forecast sensor reporting kWh *or* Wh, and the raw
# state cannot be read as kWh and left at that. Both figures also have to
# describe the same stretch of day: comparing a whole-day forecast against a
# whole-day consumption average answers a question about this morning when it
# is asked at six in the evening.
# ----------------------------------------------------------------------
def _outlook(**kwargs):
    from custom_components.omnibattery import _charge_outlook_kwh

    batteries = [
        _battery("Marstek", capacity=10.0, soc=0, limit_w=2500),
        _battery("Huawei", capacity=10.0, soc=0, limit_w=7000, dc_coupled=True),
    ]
    return _charge_outlook_kwh(_controller(batteries, **kwargs))


def test_a_forecast_in_watt_hours_is_not_taken_for_kilowatt_hours():
    """A thousandfold error, and it would call every day ample."""
    surplus_kwh, wanted = _outlook(
        forecast=30.0, unit="kWh", avg_consumption=20.0,
        produced_kwh=0.01, hours_ahead=24.0,
    )
    surplus_wh, _ = _outlook(
        forecast=30000.0, unit="Wh", avg_consumption=20.0,
        produced_kwh=0.01, hours_ahead=24.0,
    )
    assert surplus_kwh == pytest.approx(surplus_wh)
    assert wanted == pytest.approx(20.0)


def test_a_sensor_with_no_unit_at_all_is_read_as_kilowatt_hours(monkeypatch):
    """The documented configuration, and what the shared reader assumes."""
    from custom_components.omnibattery import _charge_outlook_kwh
    import custom_components.omnibattery as mod

    # At the start of the production window nothing is behind us yet.
    monkeypatch.setattr(
        mod.dt_util, "now",
        lambda: datetime(2026, 8, 25, 8, 0, tzinfo=timezone.utc),
    )
    batteries = [_battery("Marstek", capacity=10.0, soc=0, limit_w=2500)]
    controller = _controller(batteries, forecast=30.0, avg_consumption=20.0,
                             hours_ahead=0.0)
    controller.hass.states.get = lambda _eid: SimpleNamespace(state="30.0", attributes={})
    assert _charge_outlook_kwh(controller)[0] == pytest.approx(30.0)


def test_a_foreign_unit_is_refused_rather_than_guessed_at():
    """Neither kWh nor Wh: the shared reader declines, and so does the outlook."""
    assert _outlook(forecast=30.0, unit="MWh", avg_consumption=20.0) is None


def test_production_already_on_the_roof_is_taken_off_the_forecast():
    """Measured beats any curve fitted to the clock."""
    morning, _ = _outlook(
        forecast=30.0, avg_consumption=20.0, produced_kwh=0.01, hours_ahead=24.0,
    )
    afternoon, _ = _outlook(
        forecast=30.0, avg_consumption=20.0, produced_kwh=25.0, hours_ahead=6.0,
    )
    assert morning == pytest.approx(9.99, abs=0.02)
    # 5 kWh of sun left against 5 kWh of consumption still ahead.
    assert afternoon == pytest.approx(0.0, abs=0.02)
    assert afternoon < morning


def test_the_same_forecast_says_less_as_the_day_goes_on(monkeypatch):
    """Without production figures the elapsed part of the window stands in."""
    from custom_components.omnibattery import _charge_outlook_kwh
    import custom_components.omnibattery as mod

    def at(hour):
        monkeypatch.setattr(
            mod.dt_util, "now",
            lambda: datetime(2026, 8, 25, hour, 0, tzinfo=timezone.utc),
        )
        batteries = [_battery("Marstek", capacity=10.0, soc=0, limit_w=2500)]
        # Production window 08:00–20:00, consumption window the whole day.
        return _charge_outlook_kwh(_controller(
            batteries, forecast=30.0, avg_consumption=12.0, t_end=20.0,
        ))[0]

    at_eight, at_two, at_seven = at(8), at(14), at(19)
    assert at_eight > at_two > at_seven

    # At 08:00 the whole 30 kWh is still ahead, and only the twelve hours of
    # consumption up to sunset count against it — 6 of the 12 kWh daily average.
    # What the house uses after dark cannot be covered by today's sun either way.
    assert at_eight == pytest.approx(24.0, abs=0.5)
    # Half the production window gone: 15 kWh left against 3 kWh still to use.
    assert at_two == pytest.approx(12.0, abs=0.5)
    # An hour before sunset there is almost nothing left to plan with.
    assert at_seven == pytest.approx(2.0, abs=0.5)


def test_no_forecast_sensor_yields_no_opinion():
    assert _outlook(forecast=None) is None


def test_an_unreadable_forecast_yields_no_opinion():
    from custom_components.omnibattery import _charge_outlook_kwh

    controller = _controller([_battery("A", capacity=10.0, soc=0, limit_w=1000)],
                             forecast=30.0)
    controller.hass.states.get = lambda _eid: SimpleNamespace(
        state="unavailable", attributes={}
    )
    assert _charge_outlook_kwh(controller) is None


def test_a_remaining_forecast_sensor_is_taken_as_it_stands():
    """It already answers the question; subtracting production would count the
    morning twice, and the integration nags for exactly this sensor."""
    from custom_components.omnibattery import _charge_outlook_kwh

    batteries = [_battery("Marstek", capacity=10.0, soc=0, limit_w=2500)]
    controller = _controller(
        batteries, forecast=8.0, avg_consumption=12.0,
        produced_kwh=15.0, hours_ahead=0.0,
    )
    controller.solar_forecast_remaining_sensor = "sensor.remaining"
    controller.hass.states.get = lambda eid: SimpleNamespace(
        state="8.0" if eid == "sensor.remaining" else "23.0",
        attributes={"unit_of_measurement": "kWh"},
    )
    # 15 kWh already produced today is not deducted from the 8 kWh still to come.
    assert _charge_outlook_kwh(controller)[0] == pytest.approx(8.0)


def test_a_remaining_sensor_in_watt_hours_is_converted_too():
    from custom_components.omnibattery import _charge_outlook_kwh

    batteries = [_battery("Marstek", capacity=10.0, soc=0, limit_w=2500)]
    controller = _controller(
        batteries, forecast=8.0, avg_consumption=12.0, hours_ahead=0.0,
    )
    controller.solar_forecast_remaining_sensor = "sensor.remaining"
    controller.hass.states.get = lambda eid: SimpleNamespace(
        state="8000", attributes={"unit_of_measurement": "Wh"}
    )
    assert _charge_outlook_kwh(controller)[0] == pytest.approx(8.0)


def test_without_a_remaining_sensor_the_whole_day_figure_is_trimmed():
    """The legacy path, which is what most installations still have."""
    from custom_components.omnibattery import _charge_outlook_kwh

    batteries = [_battery("Marstek", capacity=10.0, soc=0, limit_w=2500)]
    controller = _controller(
        batteries, forecast=23.0, avg_consumption=12.0,
        produced_kwh=15.0, hours_ahead=0.0,
    )
    assert _charge_outlook_kwh(controller)[0] == pytest.approx(8.0)
