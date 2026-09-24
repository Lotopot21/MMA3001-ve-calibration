"""Tests for the forward engine model."""

import numpy as np
import pytest

from vetuner.drive_cycle import (
    DriveCycle,
    battery_voltage,
    generate_drive_cycle,
    intake_air_temperature,
)
from vetuner.engine_model import (
    EngineGeometry,
    Injector,
    air_density,
    air_mass_per_cycle,
    cycle_period_ms,
    fuel_mass_from_pulse_width,
    pulse_width_from_fuel_mass,
    read_ve_table,
    simulate_engine,
    target_afr,
)
from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    ve_table,
)


def test_air_density_matches_known_standard_conditions():
    # Dry air at 101.325 kPa and 15 C has a density of about 1.225 kg/m^3.
    assert float(air_density(101.325, 15.0)) == pytest.approx(1.225, abs=0.002)


def test_air_density_halves_when_pressure_halves():
    assert float(air_density(50.0, 25.0)) == pytest.approx(
        0.5 * float(air_density(100.0, 25.0))
    )


def test_air_mass_scales_with_displacement():
    small = EngineGeometry(displacement_l=1.0)
    large = EngineGeometry(displacement_l=2.0)
    m_small = air_mass_per_cycle(0.9, 100.0, 25.0, small)
    m_large = air_mass_per_cycle(0.9, 100.0, 25.0, large)
    assert float(m_large) == pytest.approx(2.0 * float(m_small))


def test_pulse_width_round_trips_to_fuel_mass():
    injector = Injector()
    mass = np.array([1e-5, 3e-5, 5e-5])
    pw = pulse_width_from_fuel_mass(mass, 14.0, injector)
    recovered = fuel_mass_from_pulse_width(pw, 14.0, injector)
    assert np.allclose(recovered, mass)


def test_pulse_width_always_exceeds_deadtime():
    injector = Injector()
    pw = pulse_width_from_fuel_mass(1e-6, 13.6, injector)
    assert float(pw) > injector.deadtime_ms_at_14v


def test_lower_voltage_lengthens_deadtime():
    injector = Injector()
    pw_low = pulse_width_from_fuel_mass(3e-5, 12.0, injector)
    pw_high = pulse_width_from_fuel_mass(3e-5, 14.0, injector)
    assert float(pw_low) > float(pw_high)


def test_cycle_period_halves_when_speed_doubles():
    engine = EngineGeometry()
    assert float(cycle_period_ms(6000.0, engine)) == pytest.approx(
        0.5 * float(cycle_period_ms(3000.0, engine))
    )


def test_target_afr_is_stoichiometric_at_light_load():
    assert float(target_afr(30.0)) == pytest.approx(14.7)


def test_target_afr_enriches_at_full_load():
    assert float(target_afr(100.0)) == pytest.approx(12.8)


def test_table_read_returns_node_values_at_breakpoints():
    table = ve_table()
    

    value = read_ve_table(table, DEFAULT_RPM_AXIS[3], DEFAULT_MAP_AXIS[5])
    assert float(value) == pytest.approx(table[3, 5])


def test_table_read_clamps_outside_axes():
    table = ve_table()
    below = read_ve_table(table, 100.0, 5.0)
    assert float(below) == pytest.approx(table[0, 0])


def test_mismatched_table_shape_raises():
    with pytest.raises(ValueError):
        read_ve_table(np.zeros((3, 3)), 3000.0, 60.0)


def test_perfect_table_is_exact_at_table_breakpoints():
    """At grid nodes there is no interpolation, so fuelling is exact."""
    rpm_grid, map_grid = np.meshgrid(DEFAULT_RPM_AXIS, DEFAULT_MAP_AXIS, indexing="ij")
    rpm = rpm_grid.ravel()
    map_kpa = map_grid.ravel()

    cycle = DriveCycle(
        time_s=np.arange(len(rpm), dtype=float) * 0.02,
        rpm=rpm,
        tps=np.full_like(rpm, 0.5),
        map_kpa=map_kpa,
        iat_c=intake_air_temperature(map_kpa),
        batt_v=battery_voltage(rpm),
    )

    result = simulate_engine(cycle, ve_table(params=TRUE_SURFACE))
    assert np.allclose(result.afr_actual, result.afr_target, rtol=1e-12)


def test_table_discretisation_error_is_small_within_the_table_domain():
    """Bilinear interpolation of a curved surface loses accuracy between nodes.

    This sets the irreducible error floor: no calibration method can do better
    than the table resolution allows. Samples below the lowest load breakpoint
    are excluded, since the error there comes from domain clamping rather than
    interpolation.
    """
    cycle = generate_drive_cycle()
    result = simulate_engine(cycle, ve_table(params=TRUE_SURFACE))
    in_domain = (~result.fuel_cut) & (result.map_kpa >= DEFAULT_MAP_AXIS[0])
    relative = np.abs(
        result.afr_actual[in_domain] / result.afr_target[in_domain] - 1.0
    )

    assert relative.max() > 0.0
    assert relative.max() < 0.01


def test_below_table_domain_causes_clamping_error():
    """Operating below the lowest load breakpoint produces a large fuelling error.

    The table read clamps to its edge node, so the ECU applies a VE from a
    different operating point. A calibration pipeline cannot correct cells it
    has no axis for, so these samples must be rejected rather than used.
    """
    cycle = generate_drive_cycle()
    result = simulate_engine(cycle, ve_table(params=TRUE_SURFACE))
    below = (~result.fuel_cut) & (result.map_kpa < DEFAULT_MAP_AXIS[0])

    assert below.any()
    relative = np.abs(result.afr_actual[below] / result.afr_target[below] - 1.0)
    assert relative.max() > 0.05


def test_afr_error_equals_ve_error():
    """AFR_measured / AFR_target must equal VE_true / VE_commanded."""
    cycle = generate_drive_cycle()
    result = simulate_engine(cycle, ve_table(params=BASE_MAP_SURFACE))
    fuelled = ~result.fuel_cut
    afr_ratio = result.afr_actual[fuelled] / result.afr_target[fuelled]
    ve_ratio = result.ve_true[fuelled] / result.ve_commanded[fuelled]
    assert np.allclose(afr_ratio, ve_ratio, rtol=1e-9)


def test_under_reading_table_runs_lean():
    cycle = generate_drive_cycle()
    result = simulate_engine(cycle, 0.8 * ve_table(params=TRUE_SURFACE))
    fuelled = ~result.fuel_cut
    assert np.all(result.afr_actual[fuelled] > result.afr_target[fuelled])


def test_fuel_cut_produces_no_fuel_and_undefined_afr():
    cycle = generate_drive_cycle()
    result = simulate_engine(cycle, ve_table(params=BASE_MAP_SURFACE))
    assert result.fuel_cut.any()
    assert np.all(result.fuel_mass_kg[result.fuel_cut] == 0.0)
    assert np.all(np.isnan(result.afr_actual[result.fuel_cut]))


def test_injector_never_saturates_on_default_cycle():
    cycle = generate_drive_cycle()
    injector = Injector()
    result = simulate_engine(cycle, ve_table(params=TRUE_SURFACE), injector=injector)
    assert result.injector_duty.max() < injector.max_duty_fraction


def test_negative_fuel_mass_raises():
    with pytest.raises(ValueError):
        pulse_width_from_fuel_mass(-1e-5, 14.0, Injector())