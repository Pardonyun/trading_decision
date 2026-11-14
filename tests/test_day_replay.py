"""Observable single-day replay behavior for the known-price example."""

import unittest
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import date, timedelta
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


def prices(day: date, first_price: int) -> tuple[HourlyPrice, ...]:
    return tuple(
        HourlyPrice(
            operating_day=day,
            platform_period=period,
            price=Decimal(first_price + period - 1),
        )
        for period in range(1, 25)
    )


class DayReplayTests(unittest.TestCase):
    def test_complete_day_uses_d_minus_7_to_make_a_valid_trading_plan(self) -> None:
        result = replay_day(
            DAY,
            prices(DAY - timedelta(days=7), 30)[::-1],
            prices(DAY, 50)[::-1],
            Decimal("0.05"),
        )

        self.assertEqual(result.decision.operating_day, DAY)
        self.assertEqual(result.decision.forecast_source_day, DAY - timedelta(days=7))
        self.assertEqual(result.decision.forecast_prices, tuple(Decimal(i) for i in range(30, 54)))
        self.assertEqual(result.settlement_prices, tuple(Decimal(i) for i in range(50, 74)))
        self.assertEqual(len(result.decision.power_mw), 24)
        self.assertLess(result.decision.power_mw[0], 0)
        self.assertGreater(result.decision.power_mw[-1], 0)
        self.assertTrue(result.decision.plan_valid)
        self.assertEqual(result.decision.solver_status, "optimal")
        self.assertIsNone(result.decision.fallback_reason)
        self.assertEqual(result.decision.starting_soc, Decimal("0.05"))
        self.assertGreater(result.daily_revenue, Decimal(0))
        self.assertEqual(result.next_starting_soc, result.decision.ending_soc)

    def test_other_history_and_changed_actual_prices_do_not_change_frozen_decision(self) -> None:
        source_day = DAY - timedelta(days=7)
        history = prices(source_day, 30) + prices(DAY - timedelta(days=1), 900)
        first = replay_day(
            DAY, history, prices(DAY, 50), Decimal("0.05")
        )
        second = replay_day(
            DAY, history, prices(DAY, -100), Decimal("0.05")
        )

        self.assertEqual(first.decision, second.decision)
        self.assertEqual(first.decision.forecast_prices, tuple(Decimal(i) for i in range(30, 54)))

    def test_missing_or_duplicate_platform_period_is_rejected(self) -> None:
        history = prices(DAY - timedelta(days=7), 30)
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            freeze_decision(DAY, history[:-1], Decimal("0.05"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_decision(DAY, history[:-1] + history[:1], Decimal("0.05"))

    def test_settlement_uses_actual_prices_and_rechecks_frozen_power(self) -> None:
        frozen = freeze_decision(
            DAY, prices(DAY - timedelta(days=7), 30), Decimal("0.05")
        )
        traded = replace(
            frozen,
            power_mw=(Decimal("-1"), Decimal("0.7225")) + (Decimal(0),) * 22,
            ending_soc=Decimal("0.05"),
        )
        actual = list(prices(DAY, 0))
        actual[0] = replace(actual[0], price=Decimal(30))
        actual[1] = replace(actual[1], price=Decimal(40))
        result = settle_day(traded, actual)

        self.assertEqual(result.daily_revenue, Decimal("-1.1000"))
        self.assertEqual(result.next_starting_soc, Decimal("0.05"))

    def test_settlement_rejects_a_frozen_record_with_inconsistent_ending_soc(self) -> None:
        frozen = freeze_decision(
            DAY, prices(DAY - timedelta(days=7), 30), Decimal("0.05")
        )
        inconsistent = replace(frozen, ending_soc=Decimal("0.95"))
        with self.assertRaisesRegex(ValueError, "ending SOC"):
            settle_day(inconsistent, prices(DAY, 50))

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
        self.assertEqual(len(output["power_mw"]), 24)
        self.assertLess(output["power_mw"][0], 0)
        self.assertGreater(output["power_mw"][-1], 0)
        self.assertEqual(output["operating_day"], "2026-11-11")
        self.assertTrue(output["plan_valid"])
        self.assertEqual(output["solver_status"], "optimal")
        self.assertIsNone(output["fallback_reason"])
        self.assertEqual(output["power_decimal_places"], 4)
        self.assertEqual(len(output["hourly_ending_soc"]), 24)
        self.assertEqual(output["starting_soc"], 0.05)
        self.assertGreater(output["predicted_revenue"], 0)
        self.assertGreater(output["daily_revenue"], 0)
        self.assertEqual(output["next_starting_soc"], output["ending_soc"])

    def test_cli_json_stays_parseable_for_volatile_prices(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "known_prices.json"
        payload = json.loads(example.read_text(encoding="utf-8"))
        payload["starting_soc"] = 0.95
        payload["historical_curves"][0]["prices"] = [
            -2, 64, 121, 52, 74, 43, -74, 58, 149, 140, 107, 29,
            -31, 10, 62, 84, 83, -40, 139, -23, 11, -34, 33, -23,
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "volatile.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "trading_decision", str(input_path)],
                capture_output=True, text=True, check=True,
            )
        output = json.loads(result.stdout)
        self.assertEqual(output["solver_status"], "optimal")
        self.assertEqual(len(output["power_mw"]), 24)


if __name__ == "__main__":
    unittest.main()
