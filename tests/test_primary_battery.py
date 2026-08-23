"""Tests for the primary battery and its house-load feedforward.

The situation this exists for: a hybrid inverter running its own
self-consumption shares the meter. It removes the grid error before the PD loop
sees one, so the loop commands nothing, and the second battery never runs — the
grid looks perfect while half the storage sits idle.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.omnibattery import (
    _apply_primary_feedforward,
    _measured_house_load_w,
    _primary_coordinator,
    _primary_feedforward_candidate_w,
    _primary_feedforward_pending,
    _primary_feedforward_w,
)


def _battery(name, *, ac_power=None, battery_power=None, soc=50, available=True):
    data = {"battery_soc": soc}
    if ac_power is not None:
        data["ac_power"] = ac_power
    if battery_power is not None:
        data["battery_power"] = battery_power
    return SimpleNamespace(name=name, data=data, is_available=available)


def _controller(batteries, *, primary="", enabled=True, limit=2500, grid_sensor=None):
    return SimpleNamespace(
        coordinators=batteries,
        primary_battery=primary,
        primary_feedforward_enabled=enabled,
        solar_production_sensor=grid_sensor,
        hass=SimpleNamespace(states=SimpleNamespace(get=lambda _eid: None)),
        previous_power=0.0,
        _battery_power_limit=lambda coordinator, is_charging: limit,
    )


# ----------------------------------------------------------------------
# house load
# ----------------------------------------------------------------------
def test_the_house_load_counts_power_this_controller_never_commanded():
    """The whole point: a battery someone else is driving still feeds the house.

    On the reference installation the grid sits at 0 W while a hybrid inverter
    discharges 800 W under its own energy manager. A commanded-power estimate
    would call that a 0 W house; the measured balance sees the 800 W.
    """
    controller = _controller([
        _battery("Huawei", battery_power=-800),   # +charge / −discharge
        _battery("Marstek", ac_power=0),
    ])
    assert _measured_house_load_w(controller, 0.0) == 800.0


def test_a_battery_that_dropped_out_is_left_out():
    """Its last reading is frozen, and the meter already shows its load."""
    controller = _controller([
        _battery("Huawei", battery_power=-800, available=False),
        _battery("Marstek", ac_power=200),
    ])
    assert _measured_house_load_w(controller, 600.0) == 800.0


def test_no_readable_battery_means_no_answer_rather_than_zero():
    controller = _controller([_battery("Marstek", available=False)])
    assert _measured_house_load_w(controller, 500.0) is None


def test_export_does_not_produce_a_negative_house():
    controller = _controller([_battery("Marstek", ac_power=1000)])
    assert _measured_house_load_w(controller, -3000.0) == 0.0


# ----------------------------------------------------------------------
# nominating the primary
# ----------------------------------------------------------------------
def test_no_primary_nominated_means_the_feature_is_inert():
    controller = _controller([_battery("Marstek", ac_power=800)], primary="")
    assert _primary_coordinator(controller) is None
    assert _primary_feedforward_w(controller, 0.0) == 0.0


def test_a_name_that_matches_nothing_is_not_guessed_at():
    controller = _controller([_battery("Marstek", ac_power=800)], primary="Venus 9")
    assert _primary_coordinator(controller) is None


def test_the_switch_gates_the_command_but_not_the_reading():
    """The figure has to be inspectable before it is acted on."""
    controller = _controller(
        [_battery("Huawei", battery_power=-800), _battery("Marstek", ac_power=0)],
        primary="Marstek",
        enabled=False,
    )
    assert _primary_feedforward_candidate_w(controller, 0.0) == 800.0
    assert _primary_feedforward_w(controller, 0.0) == 0.0


def test_a_battery_that_cannot_discharge_gets_no_feedforward():
    """min SOC, discharge disallowed, a driver limit of zero — all the same here."""
    controller = _controller(
        [_battery("Marstek", ac_power=0)], primary="Marstek", limit=0
    )
    assert _primary_feedforward_w(controller, 500.0) == 0.0


def test_the_feedforward_never_exceeds_what_the_battery_can_deliver():
    controller = _controller(
        [_battery("Huawei", battery_power=-4000), _battery("Marstek", ac_power=0)],
        primary="Marstek",
        limit=2500,
    )
    assert _primary_feedforward_w(controller, 0.0) == 2500.0


# ----------------------------------------------------------------------
# acting on it
# ----------------------------------------------------------------------
def test_the_command_is_floored_at_the_house_load():
    controller = _controller(
        [_battery("Huawei", battery_power=-800), _battery("Marstek", ac_power=0)],
        primary="Marstek",
    )
    # Discharge is negative. An idle loop is raised to cover the house...
    assert _apply_primary_feedforward(controller, 0, 0.0) == -800.0
    # ...and so is a charge command, which would otherwise fight it.
    assert _apply_primary_feedforward(controller, 500, 0.0) == -800.0


def test_a_larger_discharge_is_left_alone():
    """It is a floor, not a target: the PD may always ask for more."""
    controller = _controller(
        [_battery("Huawei", battery_power=-800), _battery("Marstek", ac_power=0)],
        primary="Marstek",
    )
    assert _apply_primary_feedforward(controller, -2000, 0.0) == -2000


def test_the_deadband_shortcut_is_skipped_while_the_floor_is_unmet():
    """A grid on target is exactly the state this feature exists to change.

    It is on target *because* the other regulator carried the load, so the
    deadband's reasoning — nothing to correct — does not apply.
    """
    controller = _controller(
        [_battery("Huawei", battery_power=-800), _battery("Marstek", ac_power=0)],
        primary="Marstek",
    )
    assert _primary_feedforward_pending(controller, 0.0) is True

    # Once the primary carries it, the shortcut is welcome again.
    controller.previous_power = -800.0
    assert _primary_feedforward_pending(controller, 0.0) is False


def test_a_small_shortfall_does_not_reopen_the_cycle():
    """Below the tolerance a correction is not worth a write."""
    controller = _controller(
        [_battery("Huawei", battery_power=-850), _battery("Marstek", ac_power=0)],
        primary="Marstek",
    )
    controller.previous_power = -800.0
    assert _primary_feedforward_pending(controller, 0.0) is False


def test_the_switch_being_off_leaves_the_deadband_untouched():
    controller = _controller(
        [_battery("Huawei", battery_power=-800)], primary="Huawei", enabled=False
    )
    assert _primary_feedforward_pending(controller, 0.0) is False


# ----------------------------------------------------------------------
# discharge order
# ----------------------------------------------------------------------
def _selector(batteries, primary=""):
    from custom_components.omnibattery.control.power_distribution import PowerDistribution

    controller = _controller(batteries, primary=primary)
    controller._active_charge_batteries = []
    controller._active_discharge_batteries = []
    selector = PowerDistribution.__new__(PowerDistribution)
    selector._controller = controller
    selector._is_battery_manual_owned = lambda _c: False
    return selector


def test_discharge_normally_starts_with_the_fullest():
    marstek = _battery("Marstek", soc=55)
    huawei = _battery("Huawei", soc=40)
    selector = _selector([huawei, marstek])
    order = selector._ordered_batteries_for_operation([huawei, marstek], False)
    assert [b.name for b in order] == ["Marstek", "Huawei"]


def test_a_nominated_primary_goes_first_even_when_emptier():
    marstek = _battery("Marstek", soc=20)
    huawei = _battery("Huawei", soc=90)
    selector = _selector([huawei, marstek], primary="Marstek")
    order = selector._ordered_batteries_for_operation([huawei, marstek], False)
    assert [b.name for b in order] == ["Marstek", "Huawei"]


def test_charging_still_fills_the_emptiest_first():
    """Otherwise the primary would be driven to both extremes and never level out."""
    marstek = _battery("Marstek", soc=80)
    huawei = _battery("Huawei", soc=30)
    selector = _selector([huawei, marstek], primary="Marstek")
    order = selector._ordered_batteries_for_operation([huawei, marstek], True)
    assert [b.name for b in order] == ["Huawei", "Marstek"]


def test_both_new_entities_are_named_in_every_language():
    import glob
    import json

    for path in ["custom_components/omnibattery/strings.json"] + sorted(
        glob.glob("custom_components/omnibattery/translations/*.json")
    ):
        entity = json.load(open(path, encoding="utf-8"))["entity"]
        assert entity["select"]["primary_battery"]["name"], path
        # The "automatic" option is a state, not a battery name, so it needs one.
        assert entity["select"]["primary_battery"]["state"]["automatic"], path
        assert entity["switch"]["primary_feedforward"]["name"], path


def test_the_panel_offers_both_controls_in_every_language():
    """A setting nobody can find is a setting nobody uses."""
    import re

    panel = open(
        "custom_components/omnibattery/frontend/marstek-panel.js", encoding="utf-8"
    ).read()

    assert '{ key: "primary_battery", domain: "select"' in panel
    assert '{ key: "primary_feedforward", domain: "switch"' in panel
    # One label set and one help text per language block.
    for key in ("secPrimary", "itemPrimaryBattery", "itemPrimaryFeedforward"):
        assert len(re.findall(r"\b%s:" % key, panel)) == 6, key
    for key in ("primary_battery", "primary_feedforward"):
        assert len(re.findall(r"^    %s: \"" % key, panel, re.M)) == 6, key
