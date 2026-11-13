"""Replay one unambiguous 24-period operating day in information-time order."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .storage_optimizer import OptimizationResult, SolverStatus, optimize_plan
from .storage_rules import (
    CAPACITY_MWH, CHARGE_EFFICIENCY, DEFAULT_POWER_DECIMAL_PLACES,
    DISCHARGE_EFFICIENCY, MAX_POWER_MW, MAX_SOC, MIN_SOC, power_step,
)


BEIJING_OFFSET = timedelta(hours=8)


def _utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value.astimezone(timezone.utc)


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
    market_interval_start: datetime
    price: Decimal
    published_at: datetime
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.operating_day, date) or isinstance(self.operating_day, datetime):
            raise ValueError("operating_day must be a date")
        if isinstance(self.platform_period, bool) or not isinstance(self.platform_period, int):
            raise ValueError("platform_period must be an integer")
        if not 1 <= self.platform_period <= 24:
            raise ValueError("platform_period must be in 1..24")
        _utc(self.market_interval_start)
        _utc(self.published_at)
        object.__setattr__(self, "price", _decimal(self.price))
        if not self.source.strip():
            raise ValueError("price source is required")


@dataclass(frozen=True)
class PlanValidation:
    ending_soc: Decimal
    hourly_ending_soc: tuple[Decimal, ...]


@dataclass(frozen=True)
class FrozenDecision:
    operating_day: date
    deadline_beijing: datetime
    data_visible_before_utc: datetime
    latest_historical_publication_utc: datetime
    forecast_source_day: date
    forecast_prices: tuple[Decimal, ...]
    power_mw: tuple[Decimal, ...]
    starting_soc: Decimal
    plan_valid: bool
    ending_soc: Decimal
    ignored_after_cutoff_count: int
    power_decimal_places: int
    solver_status: SolverStatus
    solver_predicted_revenue: Decimal | None
    predicted_revenue: Decimal
    fallback_reason: str | None


@dataclass(frozen=True)
class DayResult:
    decision: FrozenDecision
    settlement_prices: tuple[Decimal, ...]
    latest_actual_publication_utc: datetime
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
    try:
        market_timezone = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError as error:
        raise ValueError("America/Los_Angeles time-zone data is required (install tzdata)") from error
    for period, row in enumerate(ordered, 1):
        start = row.market_interval_start
        california_start = _utc(start).astimezone(market_timezone)
        if (california_start.replace(tzinfo=None), california_start.utcoffset()) != (
            start.replace(tzinfo=None), start.utcoffset()
        ):
            raise ValueError(f"{day} period {period} has the wrong California UTC offset")
        if (start.date(), start.hour, start.minute, start.second, start.microsecond) != (
            day, period - 1, 0, 0, 0
        ):
            raise ValueError(f"{day} period {period} has an ambiguous market hour")
    for previous, current in zip(ordered, ordered[1:]):
        if _utc(current.market_interval_start) - _utc(previous.market_interval_start) != timedelta(hours=1):
            raise ValueError(f"{day} market intervals are not consecutive UTC hours")
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
    deadline_beijing: datetime,
    historical_prices: Sequence[HourlyPrice],
    starting_soc: Decimal,
    *,
    power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
    optimizer: Callable[[tuple[Decimal, ...], Decimal], OptimizationResult] | None = None,
) -> FrozenDecision:
    """Freeze a formatted, independently checked D−7 forecast plan."""
    if deadline_beijing.utcoffset() != BEIJING_OFFSET:
        raise ValueError("deadline must be in Beijing time (+08:00)")
    cutoff_utc = _utc(deadline_beijing)
    source_day = operating_day - timedelta(days=7)
    source_records = tuple(row for row in historical_prices if row.operating_day == source_day)
    visible_records = tuple(row for row in source_records if _utc(row.published_at) < cutoff_utc)
    ordered = _ordered_curve(visible_records, source_day)
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
        deadline_beijing=deadline_beijing,
        data_visible_before_utc=cutoff_utc,
        latest_historical_publication_utc=max(_utc(row.published_at) for row in ordered),
        forecast_source_day=source_day,
        forecast_prices=forecast_prices,
        power_mw=plan,
        starting_soc=initial_soc,
        plan_valid=True,
        ending_soc=validation.ending_soc,
        ignored_after_cutoff_count=len(source_records) - len(visible_records),
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
        latest_actual_publication_utc=max(_utc(row.published_at) for row in ordered),
        daily_revenue=revenue,
        next_starting_soc=validation.ending_soc,
    )


def replay_day(
    operating_day: date,
    deadline_beijing: datetime,
    historical_prices: Sequence[HourlyPrice],
    actual_prices: Sequence[HourlyPrice],
    starting_soc: Decimal = Decimal("0.05"),
    *,
    power_decimal_places: int = DEFAULT_POWER_DECIMAL_PLACES,
    optimizer: Callable[[tuple[Decimal, ...], Decimal], OptimizationResult] | None = None,
) -> DayResult:
    """Replay the decision first, then reveal actual prices for settlement."""
    decision = freeze_decision(
        operating_day, deadline_beijing, historical_prices, starting_soc,
        power_decimal_places=power_decimal_places, optimizer=optimizer,
    )
    return settle_day(decision, actual_prices)
