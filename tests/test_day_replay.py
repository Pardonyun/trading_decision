"""Observable single-day replay behavior for the known-price example."""

import unittest
import json
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trading_decision.day_replay import (
    HourlyPrice,
    freeze_decision,
    replay_day,
    settle_day,
    validate_plan,
)


DAY = date(2026, 11, 11)
DEADLINE = datetime.fromisoformat("2026-11-11T02:00:00+08:00")
HISTORY_PUBLICATION = datetime.fromisoformat("2026-11-04T21:00:00+00:00")
ACTUAL_PUBLICATION = datetime.fromisoformat("2026-11-10T21:00:00+00:00")


def prices(
    day: date, published_at: datetime, first_price: int, offset: str = "-08:00"
) -> tuple[HourlyPrice, ...]:
    start = datetime.fromisoformat(f"{day}T00:00:00{offset}")
    return tuple(
        HourlyPrice(
            operating_day=day,
            platform_period=period,
            market_interval_start=start + timedelta(hours=period - 1),
            price=Decimal(first_price + period - 1),
            published_at=published_at,
            source="known-price-example",
        )
        for period in range(1, 25)
    )


class DayReplayTests(unittest.TestCase):
    def test_complete_day_uses_d_minus_7_and_keeps_initial_soc(self) -> None:
        result = replay_day(
            DAY,
            DEADLINE,
            prices(DAY - timedelta(days=7), HISTORY_PUBLICATION, 30),
            prices(DAY, ACTUAL_PUBLICATION, 50),
            Decimal("0.05"),
        )

        self.assertEqual(result.decision.operating_day, DAY)
        self.assertEqual(result.decision.forecast_source_day, DAY - timedelta(days=7))
        self.assertEqual(result.decision.forecast_prices, tuple(Decimal(i) for i in range(30, 54)))
        self.assertEqual(result.decision.power_mw, (Decimal(0),) * 24)
        self.assertTrue(result.decision.plan_valid)
        self.assertEqual(result.decision.starting_soc, Decimal("0.05"))
        self.assertEqual(result.daily_revenue, Decimal(0))
        self.assertEqual(result.next_starting_soc, Decimal("0.05"))
        self.assertEqual(result.decision.data_visible_before_utc.isoformat(), "2026-11-10T18:00:00+00:00")

    def test_late_history_and_changed_actual_prices_do_not_change_frozen_decision(self) -> None:
        source_day = DAY - timedelta(days=7)
        timely = prices(source_day, HISTORY_PUBLICATION, 30)
        late = prices(source_day, ACTUAL_PUBLICATION, 900)
        frozen = freeze_decision(DAY, DEADLINE, timely + late, Decimal("0.05"))
        first = settle_day(frozen, prices(DAY, ACTUAL_PUBLICATION, 50))
        second = settle_day(frozen, prices(DAY, ACTUAL_PUBLICATION, -100))

        self.assertEqual(first.decision, second.decision)
        self.assertEqual(frozen.forecast_prices, tuple(Decimal(i) for i in range(30, 54)))
        self.assertEqual(frozen.ignored_after_cutoff_count, 24)
        self.assertEqual(first.daily_revenue, second.daily_revenue)

    def test_price_published_at_the_deadline_is_not_visible(self) -> None:
        source_day = DAY - timedelta(days=7)
        history = list(prices(source_day, HISTORY_PUBLICATION, 30))
        history[0] = replace(history[0], published_at=DEADLINE)
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            freeze_decision(DAY, DEADLINE, history, Decimal("0.05"))

    def test_platform_cutoff_can_fall_on_the_previous_beijing_day(self) -> None:
        cutoff = datetime.fromisoformat("2026-11-10T23:00:00+08:00")
        frozen = freeze_decision(
            DAY, cutoff, prices(DAY - timedelta(days=7), HISTORY_PUBLICATION, 30), Decimal("0.05")
        )
        self.assertEqual(frozen.data_visible_before_utc.isoformat(), "2026-11-10T15:00:00+00:00")

    def test_missing_or_duplicate_visible_hour_is_rejected(self) -> None:
        history = prices(DAY - timedelta(days=7), HISTORY_PUBLICATION, 30)
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            freeze_decision(DAY, DEADLINE, history[:-1], Decimal("0.05"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_decision(DAY, DEADLINE, history[:-1] + history[:1], Decimal("0.05"))

    def test_market_hours_must_have_an_unambiguous_24_hour_mapping(self) -> None:
        history = prices(DAY - timedelta(days=7), HISTORY_PUBLICATION, 30)
        shifted = replace(history[2], market_interval_start=history[2].market_interval_start + timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "ambiguous market hour"):
            freeze_decision(DAY, DEADLINE, history[:2] + (shifted,) + history[3:], Decimal("0.05"))

    def test_california_daylight_saving_offset_must_match_the_market_day(self) -> None:
        summer_day = date(2026, 7, 15)
        deadline = datetime.fromisoformat("2026-07-15T02:00:00+08:00")
        source_day = summer_day - timedelta(days=7)
        publication = datetime.fromisoformat("2026-07-08T21:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "California UTC offset"):
            freeze_decision(
                summer_day, deadline, prices(source_day, publication, 30), Decimal("0.05")
            )
        correct = prices(source_day, publication, 30, "-07:00")
        self.assertEqual(len(freeze_decision(summer_day, deadline, correct, Decimal("0.05")).forecast_prices), 24)

    def test_settlement_uses_actual_prices_and_rechecks_frozen_power(self) -> None:
        frozen = freeze_decision(
            DAY, DEADLINE, prices(DAY - timedelta(days=7), HISTORY_PUBLICATION, 30), Decimal("0.05")
        )
        traded = replace(
            frozen,
            power_mw=(Decimal("-1"), Decimal("0.7225")) + (Decimal(0),) * 22,
        )
        actual = list(prices(DAY, ACTUAL_PUBLICATION, 0))
        actual[0] = replace(actual[0], price=Decimal(30))
        actual[1] = replace(actual[1], price=Decimal(40))
        result = settle_day(traded, actual)

        self.assertEqual(result.daily_revenue, Decimal("-1.1000"))
        self.assertEqual(result.next_starting_soc, Decimal("0.05"))

    def test_last_hour_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "period 24"):
            validate_plan((Decimal(0),) * 23 + (Decimal("-1"),), Decimal("0.95"))

    def test_checked_in_example_is_repeatable_from_cli(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "known_prices.json"
        command = [sys.executable, "-m", "trading_decision", str(example)]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        output = json.loads(first.stdout)

        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(output["forecast_prices"], list(range(30, 54)))
        self.assertEqual(output["power_mw"], [0] * 24)
        self.assertEqual(output["daily_revenue"], 0)
        self.assertEqual(output["next_starting_soc"], 0.05)


if __name__ == "__main__":
    unittest.main()
