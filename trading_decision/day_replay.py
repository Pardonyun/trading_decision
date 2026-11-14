"""Replay one 24-period operating day from manually supplied price curves."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Sequence

from .storage_optimizer import OptimizationResult, SolverStatus, optimize_plan
from .storage_rules import (
    CAPACITY_MWH, CHARGE_EFFICIENCY, DEFAULT_POWER_DECIMAL_PLACES,
    DISCHARGE_EFFICIENCY, MAX_POWER_MW, MAX_SOC, MIN_SOC, power_step,
)


def _decimal(value: Decimal | int | str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("numeric values cannot be booleans")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("numeric values must be finite")
    return number


@dataclass(frozen=True)
class HourlyPrice:
    operating_day: date
    platform_period: int
    price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.operating_day, date) or isinstance(self.operating_day, datetime):
            raise ValueError("operating_day must be a date")
        if isinstance(self.platform_period, bool) or not isinstance(self.platform_period, int):
            raise ValueError("platform_period must be an integer")
        if not 1 <= self.platform_period <= 24:
            raise ValueError("platform_period must be in 1..24")
        object.__setattr__(self, "price", _decimal(self.price))


@dataclass(frozen=True)
class PlanValidation:
    ending_soc: Decimal
    hourly_ending_soc: tuple[Decimal, ...]


@dataclass(frozen=True)
class FrozenDecision:
    operating_day: date
    forecast_source_day: date
    forecast_prices: tuple[Decimal, ...]
    power_mw: tuple[Decimal, ...]
    starting_soc: Decimal
    plan_valid: bool
    ending_soc: Decimal
    power_decimal_places: int
    solver_status: SolverStatus
    solver_predicted_revenue: Decimal | None
    predicted_revenue: Decimal
    fallback_reason: str | None


@dataclass(frozen=True)
class DayResult:
    decision: FrozenDecision
    settlement_prices: tuple[Decimal, ...]
    daily_revenue: Decimal
    next_starting_soc: Decimal


def _ordered_curve(records: Sequence[HourlyPrice], day: date) -> tuple[HourlyPrice, ...]:
    if len(records) != 24:
        raise ValueError(f"{day} requires exactly 24 hourly prices, got {len(records)}")
    ordered = tuple(sorted(records, key=lambda row: row.platform_period))
    if tuple(row.platform_period for row in ordered) != tuple(range(1, 25)):
        raise ValueError(f"{day} has missing or duplicate platform periods")
    if any(row.operating_day != day for row in ordered):
        raise ValueError("price operating day does not match the expected day")
    return ordered


def validate_plan(
    power_mw: Sequence[Decimal], starting_soc: Decimal,
    power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
) -> PlanValidation:
    """Recalculate every submitted hour, including the 24th hour's ending SOC."""
    submission_step = power_step(power_decimal_places)
    initial_soc = _decimal(starting_soc)
    if not MIN_SOC <= initial_soc <= MAX_SOC:
        raise ValueError("starting SOC must be between 5% and 95%")
    if len(power_mw) != 24:
        raise ValueError("a storage plan requires exactly 24 power values")
    energy_mwh = CAPACITY_MWH * initial_soc
    hourly_ending_soc: list[Decimal] = []
    for period, value in enumerate(power_mw, 1):
        power = _decimal(value)
        if power != power.quantize(submission_step):
            raise ValueError(f"period {period} exceeds the submission precision")
        if abs(power) > MAX_POWER_MW:
            raise ValueError(f"period {period} exceeds the 7 MW power limit")
        if power < 0:
            energy_mwh += -power * CHARGE_EFFICIENCY
        else:
            energy_mwh -= power / DISCHARGE_EFFICIENCY
        if not CAPACITY_MWH * MIN_SOC <= energy_mwh <= CAPACITY_MWH * MAX_SOC:
            raise ValueError(f"period {period} breaches the SOC boundary")
        hourly_ending_soc.append(energy_mwh / CAPACITY_MWH)
    return PlanValidation(
        ending_soc=energy_mwh / CAPACITY_MWH,
        hourly_ending_soc=tuple(hourly_ending_soc),
    )


def freeze_decision(
    operating_day: date,
    historical_prices: Sequence[HourlyPrice],
    starting_soc: Decimal,
    *,
    power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
    optimizer: Callable[[tuple[Decimal, ...], Decimal], OptimizationResult] | None = None,
) -> FrozenDecision:
    """Freeze a D−7 baseline plan using a supplied historical price curve."""
    source_day = operating_day - timedelta(days=7)
    source_records = tuple(row for row in historical_prices if row.operating_day == source_day)
    ordered = _ordered_curve(source_records, source_day)
    forecast_prices = tuple(row.price for row in ordered)
    initial_soc = _decimal(starting_soc)
    zero_plan = (Decimal(0),) * 24
    validate_plan(zero_plan, initial_soc, power_decimal_places)
    solver_status: SolverStatus = "error"
    solver_predicted_revenue = None
    fallback_reason = None
    plan: tuple[Decimal, ...]
    try:
        solution = (
            optimize_plan(forecast_prices, initial_soc, power_decimal_places=power_decimal_places)
            if optimizer is None else optimizer(forecast_prices, initial_soc)
        )
        solver_status = solution.status
        solver_predicted_revenue = solution.predicted_revenue
        if solution.status != "optimal" or solution.power_mw is None:
            raise ValueError(solution.reason or "solver returned no usable solution")
        submission_step = power_step(power_decimal_places)
        candidate = tuple(
            _decimal(value).quantize(submission_step, rounding=ROUND_DOWN)
            for value in solution.power_mw
        )
        validation = validate_plan(candidate, initial_soc, power_decimal_places)
        candidate_revenue = sum(
            (price * power for price, power in zip(forecast_prices, candidate)), Decimal(0)
        )
        if candidate_revenue < 0:
            raise ValueError("formatted candidate has negative predicted revenue")
        if candidate_revenue == 0:
            plan = zero_plan
            validation = validate_plan(plan, initial_soc, power_decimal_places)
        else:
            plan = tuple(Decimal(0) if power == 0 else power for power in candidate)
    except Exception as error:
        fallback_reason = f"{type(error).__name__}: {error}"
        plan = zero_plan
        validation = validate_plan(plan, initial_soc, power_decimal_places)
    predicted_revenue = sum(
        (price * power for price, power in zip(forecast_prices, plan)), Decimal(0)
    )
    return FrozenDecision(
        operating_day=operating_day,
        forecast_source_day=source_day,
        forecast_prices=forecast_prices,
        power_mw=plan,
        starting_soc=initial_soc,
        plan_valid=True,
        ending_soc=validation.ending_soc,
        power_decimal_places=power_decimal_places,
        solver_status=solver_status,
        solver_predicted_revenue=solver_predicted_revenue,
        predicted_revenue=predicted_revenue,
        fallback_reason=fallback_reason,
    )


def settle_day(decision: FrozenDecision, actual_prices: Sequence[HourlyPrice]) -> DayResult:
    """Settle a previously frozen plan with the supplied actual prices."""
    ordered = _ordered_curve(actual_prices, decision.operating_day)
    validation = validate_plan(decision.power_mw, decision.starting_soc, decision.power_decimal_places)
    if validation.ending_soc != decision.ending_soc:
        raise ValueError("frozen ending SOC does not match the validated plan")
    actual = tuple(row.price for row in ordered)
    revenue = sum((price * power for price, power in zip(actual, decision.power_mw)), Decimal(0))
    return DayResult(
        decision=decision,
        settlement_prices=actual,
        daily_revenue=revenue,
        next_starting_soc=validation.ending_soc,
    )


def replay_day(
    operating_day: date,
    historical_prices: Sequence[HourlyPrice],
    actual_prices: Sequence[HourlyPrice],
    starting_soc: Decimal = Decimal("0.05"),
    *,
    power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
    optimizer: Callable[[tuple[Decimal, ...], Decimal], OptimizationResult] | None = None,
) -> DayResult:
    """Replay the decision first, then reveal actual prices for settlement."""
    decision = freeze_decision(
        operating_day, historical_prices, starting_soc,
        power_decimal_places=power_decimal_places, optimizer=optimizer,
    )
    return settle_day(decision, actual_prices)
