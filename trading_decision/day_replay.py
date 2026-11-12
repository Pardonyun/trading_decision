"""Replay one unambiguous 24-period operating day in information-time order."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


CAPACITY_MWH = Decimal("14")
MIN_SOC = Decimal("0.05")
MAX_SOC = Decimal("0.95")
MAX_POWER_MW = Decimal("7")
CHARGE_EFFICIENCY = Decimal("0.85")
DISCHARGE_EFFICIENCY = Decimal("0.85")
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


def validate_plan(power_mw: Sequence[Decimal], starting_soc: Decimal) -> PlanValidation:
    """Recalculate every submitted hour, including the 24th hour's ending SOC."""
    initial_soc = _decimal(starting_soc)
    if not MIN_SOC <= initial_soc <= MAX_SOC:
        raise ValueError("starting SOC must be between 5% and 95%")
    if len(power_mw) != 24:
        raise ValueError("a storage plan requires exactly 24 power values")
    energy_mwh = CAPACITY_MWH * initial_soc
    for period, value in enumerate(power_mw, 1):
        power = _decimal(value)
        if abs(power) > MAX_POWER_MW:
            raise ValueError(f"period {period} exceeds the 7 MW power limit")
        if power < 0:
            energy_mwh += -power * CHARGE_EFFICIENCY
        else:
            energy_mwh -= power / DISCHARGE_EFFICIENCY
        if not CAPACITY_MWH * MIN_SOC <= energy_mwh <= CAPACITY_MWH * MAX_SOC:
            raise ValueError(f"period {period} breaches the SOC boundary")
    return PlanValidation(ending_soc=energy_mwh / CAPACITY_MWH)


def freeze_decision(
    operating_day: date,
    deadline_beijing: datetime,
    historical_prices: Sequence[HourlyPrice],
    starting_soc: Decimal,
) -> FrozenDecision:
    """Copy the visible D−7 prices and freeze a checked zero-power plan."""
    if deadline_beijing.utcoffset() != BEIJING_OFFSET:
        raise ValueError("deadline must be in Beijing time (+08:00)")
    cutoff_utc = _utc(deadline_beijing)
    source_day = operating_day - timedelta(days=7)
    source_records = tuple(row for row in historical_prices if row.operating_day == source_day)
    visible_records = tuple(row for row in source_records if _utc(row.published_at) < cutoff_utc)
    ordered = _ordered_curve(visible_records, source_day)
    zero_plan = (Decimal(0),) * 24
    validation = validate_plan(zero_plan, starting_soc)
    return FrozenDecision(
        operating_day=operating_day,
        deadline_beijing=deadline_beijing,
        data_visible_before_utc=cutoff_utc,
        latest_historical_publication_utc=max(_utc(row.published_at) for row in ordered),
        forecast_source_day=source_day,
        forecast_prices=tuple(row.price for row in ordered),
        power_mw=zero_plan,
        starting_soc=_decimal(starting_soc),
        plan_valid=True,
        ending_soc=validation.ending_soc,
        ignored_after_cutoff_count=len(source_records) - len(visible_records),
    )


def settle_day(decision: FrozenDecision, actual_prices: Sequence[HourlyPrice]) -> DayResult:
    """Settle a previously frozen plan with the supplied actual prices."""
    ordered = _ordered_curve(actual_prices, decision.operating_day)
    validation = validate_plan(decision.power_mw, decision.starting_soc)
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
) -> DayResult:
    """Replay the decision first, then reveal actual prices for settlement."""
    decision = freeze_decision(operating_day, deadline_beijing, historical_prices, starting_soc)
    return settle_day(decision, actual_prices)
