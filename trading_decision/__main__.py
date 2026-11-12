"""Run the checked-in known-price single-day replay from JSON."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .day_replay import DayResult, HourlyPrice, freeze_decision, settle_day


def _curve_rows(curve: dict[str, Any]) -> tuple[HourlyPrice, ...]:
    starts = curve["market_interval_starts"]
    prices = curve["prices"]
    if len(starts) != 24 or len(prices) != 24:
        raise ValueError("each sample curve needs 24 market starts and 24 prices")
    day = date.fromisoformat(curve["operating_day"])
    published = datetime.fromisoformat(curve["published_at"])
    return tuple(
        HourlyPrice(
            operating_day=day,
            platform_period=period,
            market_interval_start=datetime.fromisoformat(start),
            price=Decimal(str(price)),
            published_at=published,
            source=curve["source"],
        )
        for period, (start, price) in enumerate(zip(starts, prices, strict=True), 1)
    )


def _number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _output(result: DayResult) -> dict[str, Any]:
    decision = result.decision
    return {
        "operating_day": decision.operating_day.isoformat(),
        "deadline_beijing": decision.deadline_beijing.isoformat(),
        "data_visible_before_utc": decision.data_visible_before_utc.isoformat(),
        "latest_historical_publication_utc": decision.latest_historical_publication_utc.isoformat(),
        "forecast_source_day": decision.forecast_source_day.isoformat(),
        "forecast_prices": [_number(value) for value in decision.forecast_prices],
        "power_mw": [_number(value) for value in decision.power_mw],
        "plan_valid": decision.plan_valid,
        "starting_soc": _number(decision.starting_soc),
        "ending_soc": _number(decision.ending_soc),
        "ignored_after_cutoff_count": decision.ignored_after_cutoff_count,
        "settlement_prices": [_number(value) for value in result.settlement_prices],
        "latest_actual_publication_utc": result.latest_actual_publication_utc.isoformat(),
        "daily_revenue": _number(result.daily_revenue),
        "next_starting_soc": _number(result.next_starting_soc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one known-price operating day")
    parser.add_argument("input", type=Path, help="JSON file containing a known 24-period sample")
    arguments = parser.parse_args()
    try:
        with arguments.input.open(encoding="utf-8") as source:
            payload = json.load(source, parse_float=Decimal)
        historical = tuple(
            row for curve in payload["historical_curves"] for row in _curve_rows(curve)
        )
        decision = freeze_decision(
            date.fromisoformat(payload["operating_day"]),
            datetime.fromisoformat(payload["deadline_beijing"]),
            historical,
            Decimal(str(payload["starting_soc"])),
        )
        actual = _curve_rows(payload["actual_curve"])
        result = settle_day(decision, actual)
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as error:
        parser.error(str(error))
    print(json.dumps(_output(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
