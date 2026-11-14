"""Run the checked-in known-price single-day replay from JSON."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .day_replay import DayResult, HourlyPrice, freeze_decision, settle_day, validate_plan


@contextmanager
def _quiet_native_solver_output():
    """Keep native HiGHS diagnostics from corrupting the CLI JSON stream."""
    output_fd = sys.stdout.fileno()
    saved_fd = os.dup(output_fd)
    try:
        sys.stdout.flush()
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), output_fd)
            try:
                yield
            finally:
                # HiGHS can emit stray C stdout lines even with disp=False.
                runtime = ctypes.CDLL("msvcrt") if os.name == "nt" else ctypes.CDLL(None)
                runtime.fflush(None)
                os.dup2(saved_fd, output_fd)
    finally:
        os.close(saved_fd)


def _curve_rows(curve: dict[str, Any]) -> tuple[HourlyPrice, ...]:
    prices = curve["prices"]
    if len(prices) != 24:
        raise ValueError("each sample curve needs exactly 24 prices")
    day = date.fromisoformat(curve["operating_day"])
    return tuple(
        HourlyPrice(
            operating_day=day,
            platform_period=period,
            price=Decimal(str(price)),
        )
        for period, price in enumerate(prices, 1)
    )


def _number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _output(result: DayResult) -> dict[str, Any]:
    decision = result.decision
    validation = validate_plan(
        decision.power_mw, decision.starting_soc, decision.power_decimal_places
    )
    return {
        "operating_day": decision.operating_day.isoformat(),
        "forecast_source_day": decision.forecast_source_day.isoformat(),
        "forecast_prices": [_number(value) for value in decision.forecast_prices],
        "power_mw": [_number(value) for value in decision.power_mw],
        "power_decimal_places": decision.power_decimal_places,
        "plan_valid": decision.plan_valid,
        "solver_status": decision.solver_status,
        "solver_predicted_revenue": (
            _number(decision.solver_predicted_revenue)
            if decision.solver_predicted_revenue is not None else None
        ),
        "predicted_revenue": _number(decision.predicted_revenue),
        "fallback_reason": decision.fallback_reason,
        "starting_soc": _number(decision.starting_soc),
        "ending_soc": _number(decision.ending_soc),
        "hourly_ending_soc": [_number(value) for value in validation.hourly_ending_soc],
        "settlement_prices": [_number(value) for value in result.settlement_prices],
        "daily_revenue": _number(result.daily_revenue),
        "next_starting_soc": _number(result.next_starting_soc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one known-price operating day")
    parser.add_argument("input", type=Path, help="JSON file containing a known 24-period sample")
    parser.add_argument(
        "--power-decimal-places", type=int, default=4,
        help="submission precision for power in MW (default: 4; verify platform requirement)",
    )
    arguments = parser.parse_args()
    try:
        with arguments.input.open(encoding="utf-8") as source:
            payload = json.load(source, parse_float=Decimal)
        historical = tuple(
            row for curve in payload["historical_curves"] for row in _curve_rows(curve)
        )
        with _quiet_native_solver_output():
            decision = freeze_decision(
                date.fromisoformat(payload["operating_day"]),
                historical,
                Decimal(str(payload["starting_soc"])),
                power_decimal_places=arguments.power_decimal_places,
            )
        actual = _curve_rows(payload["actual_curve"])
        result = settle_day(decision, actual)
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as error:
        parser.error(str(error))
    print(json.dumps(_output(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
