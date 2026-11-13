"""Maximize one day's forecast storage revenue with exclusive hourly modes."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Literal, Sequence

from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from .storage_rules import (
    CAPACITY_MWH, CHARGE_EFFICIENCY, DEFAULT_POWER_DECIMAL_PLACES,
    DISCHARGE_EFFICIENCY, HOURS,
    MAX_POWER_MW, MAX_SOC, MIN_SOC, power_step,
)

CHARGE_EFFICIENCY_FLOAT = float(CHARGE_EFFICIENCY)
DISCHARGE_EFFICIENCY_FLOAT = float(DISCHARGE_EFFICIENCY)
DEFAULT_TIME_LIMIT_SECONDS = 10.0
OBJECTIVE_SCALE = 10_000.0
SolverStatus = Literal["optimal", "time_limit", "infeasible", "unbounded", "error", "no_solution"]


def _submitted_power(
    result: OptimizeResult, power_index: int, step: Decimal, active_hours: int,
) -> tuple[Decimal, ...]:
    """Convert solved integer power units to exact submission values."""
    active_power = tuple(
        Decimal(round(result.x[power_index + hour])) * step
        for hour in range(active_hours)
    )
    return active_power + (Decimal(0),) * (HOURS - active_hours)


@dataclass(frozen=True)
class OptimizationResult:
    power_mw: tuple[Decimal, ...] | None
    status: SolverStatus
    reason: str | None
    predicted_revenue: Decimal | None


def _solve_status(result: OptimizeResult) -> SolverStatus:
    statuses: dict[int, SolverStatus] = {
        0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded",
    }
    return statuses.get(result.status, "error")


def optimize_plan(
    forecast_prices: Sequence[Decimal], starting_soc: Decimal,
    *, power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
) -> OptimizationResult:
    """Return a submission-precision candidate for independent validation."""
    if len(forecast_prices) != HOURS:
        raise ValueError("forecast requires exactly 24 prices")
    prices = tuple(Decimal(price) for price in forecast_prices)
    if not all(price.is_finite() for price in prices):
        raise ValueError("forecast prices must be finite")
    soc = Decimal(starting_soc)
    if not soc.is_finite() or not MIN_SOC <= soc <= MAX_SOC:
        raise ValueError("starting SOC must be between 5% and 95%")
    if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time limit must be a positive finite number")
    submission_step = power_step(power_decimal_places)
    active_hours = max((hour + 1 for hour, price in enumerate(prices) if price != 0), default=0)
    if active_hours == 0:
        return OptimizationResult((Decimal(0),) * HOURS, "optimal", None, Decimal(0))

    # The names below are starting offsets of hourly variable blocks.
    # Signed q_h is integer submission units. Exclusive charge/discharge modes
    # make the continuous c_h and d_h equal to |q_h| on the active side.
    # E_(h+1) is measured in step-sized MWh units for solver scaling.
    power_index, charge, discharge, mode, energy = (
        0, active_hours, 2 * active_hours, 3 * active_hours, 4 * active_hours
    )
    count = 5 * active_hours
    maximum_power_units = int(MAX_POWER_MW / submission_step)
    minimum_energy_units = float(CAPACITY_MWH * MIN_SOC / submission_step)
    maximum_energy_units = float(CAPACITY_MWH * MAX_SOC / submission_step)
    lower_bounds = [0.0] * count
    lower_bounds[power_index:charge] = [-float(maximum_power_units)] * active_hours
    upper_bounds = [float(maximum_power_units)] * (3 * active_hours)
    upper_bounds += [1.0] * active_hours + [maximum_energy_units] * active_hours
    lower_bounds[energy:] = [minimum_energy_units] * active_hours
    integrality = (
        [1] * active_hours + [0] * (2 * active_hours) +
        [1] * active_hours + [0] * active_hours
    )
    rows: list[list[float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []
    for hour in range(active_hours):
        balance = [0.0] * count
        balance[charge + hour] = -CHARGE_EFFICIENCY_FLOAT
        balance[discharge + hour] = 1.0 / DISCHARGE_EFFICIENCY_FLOAT
        balance[energy + hour] = 1.0
        if hour:
            balance[energy + hour - 1] = -1.0
        rhs = float(CAPACITY_MWH * soc / submission_step) if hour == 0 else 0.0
        rows.append(balance)
        row_lower.append(rhs)
        row_upper.append(rhs)

        signed_power = [0.0] * count
        signed_power[power_index + hour] = 1.0
        signed_power[charge + hour] = 1.0
        signed_power[discharge + hour] = -1.0
        rows.append(signed_power)
        row_lower.append(0.0)
        row_upper.append(0.0)

        charging = [0.0] * count
        charging[charge + hour] = 1.0
        charging[mode + hour] = -float(maximum_power_units)
        rows.append(charging)
        row_lower.append(float("-inf"))
        row_upper.append(0.0)

        discharging = [0.0] * count
        discharging[discharge + hour] = 1.0
        discharging[mode + hour] = float(maximum_power_units)
        rows.append(discharging)
        row_lower.append(float("-inf"))
        row_upper.append(float(maximum_power_units))

    price_scale = max(abs(price) for price in prices)
    scaled_prices = tuple(
        OBJECTIVE_SCALE * float(price / price_scale) for price in prices[:active_hours]
    )
    revenue_coefficients = list(scaled_prices) + [0.0] * (4 * active_hours)
    objective = [-coefficient for coefficient in revenue_coefficients]
    bounds = Bounds(lower_bounds, upper_bounds)
    constraints = LinearConstraint(rows, row_lower, row_upper)
    options = {"time_limit": time_limit_seconds, "mip_rel_gap": 0.0}
    revenue_result = milp(objective, integrality=integrality, bounds=bounds,
                          constraints=constraints, options=options)
    status = _solve_status(revenue_result)
    if status != "optimal":
        return OptimizationResult(None, status, str(revenue_result.message), None)
    if (revenue_result.x is None or revenue_result.fun is None or
            not isfinite(revenue_result.fun) or
            not all(isfinite(value) for value in revenue_result.x)):
        return OptimizationResult(None, "no_solution", "solver returned no usable optimum", None)
    maximum_scaled_revenue = -revenue_result.fun
    if maximum_scaled_revenue <= 0:
        return OptimizationResult((Decimal(0),) * HOURS, "optimal", None, Decimal(0))

    revenue_power = _submitted_power(revenue_result, power_index, submission_step, active_hours)

    # Revenue remains the primary objective. Among equivalent optima, use less
    # total charge and discharge; no wear cost is added to actual settlement.
    tolerance = 1e-7
    rows.append(revenue_coefficients)
    row_lower.append(maximum_scaled_revenue - tolerance)
    row_upper.append(float("inf"))
    throughput = (
        [0.0] * active_hours + [1.0] * (2 * active_hours) + [0.0] * (2 * active_hours)
    )
    throughput_result = milp(throughput, integrality=integrality, bounds=bounds,
                             constraints=LinearConstraint(rows, row_lower, row_upper),
                             options=options)
    status = _solve_status(throughput_result)
    if status != "optimal":
        return OptimizationResult(None, status, f"tie-break solve: {throughput_result.message}", None)
    if throughput_result.x is None or not all(isfinite(value) for value in throughput_result.x):
        return OptimizationResult(None, "no_solution", "tie-break solve returned no usable plan", None)
    throughput_power = _submitted_power(throughput_result, power_index, submission_step, active_hours)
    revenue_candidate = sum((price * value for price, value in zip(prices, revenue_power)), Decimal(0))
    throughput_candidate = sum((price * value for price, value in zip(prices, throughput_power)), Decimal(0))
    if throughput_candidate > revenue_candidate or (
        throughput_candidate == revenue_candidate and
        sum(map(abs, throughput_power)) < sum(map(abs, revenue_power))
    ):
        power, submitted_revenue = throughput_power, throughput_candidate
    else:
        power, submitted_revenue = revenue_power, revenue_candidate
    if submitted_revenue <= 0:
        power = (Decimal(0),) * HOURS
    return OptimizationResult(power, "optimal", None, revenue_candidate)
