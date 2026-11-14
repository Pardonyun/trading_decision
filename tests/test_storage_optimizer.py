"""Public behavior of the D−7 storage decision and independent plan validator."""

import unittest
from datetime import date, timedelta
from decimal import Decimal

from trading_decision.day_replay import HourlyPrice, freeze_decision, settle_day, validate_plan
from trading_decision.storage_optimizer import OptimizationResult, optimize_plan


DAY = date(2026, 11, 11)


def curve(day: date, values: tuple[int | Decimal, ...]) -> tuple[HourlyPrice, ...]:
    return tuple(
        HourlyPrice(day, period, Decimal(price))
        for period, price in enumerate(values, 1)
    )


class StorageOptimizerTests(unittest.TestCase):
    def test_zero_revenue_prefers_no_operations(self) -> None:
        result = optimize_plan((Decimal(0),) * 24, Decimal("0.05"))
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.power_mw, (Decimal(0),) * 24)
        self.assertEqual(result.predicted_revenue, Decimal(0))

    def test_small_nonzero_price_keeps_revenue_as_primary_objective(self) -> None:
        forecast = (Decimal("-0.000001"),) + (Decimal(0),) * 23
        result = optimize_plan(forecast, Decimal("0.05"))
        self.assertEqual(result.power_mw, (Decimal(-7),) + (Decimal(0),) * 23)

    def test_tiny_positive_formatted_profit_is_preferred_to_zero(self) -> None:
        sale_price = Decimal(1) / Decimal("0.7225") + Decimal("0.00000001")
        forecast = (Decimal(1), sale_price) + (Decimal(0),) * 22
        result = optimize_plan(forecast, Decimal("0.05"))
        self.assertEqual(result.status, "optimal")
        assert result.power_mw is not None
        self.assertEqual(result.power_mw[:2], (Decimal(-7), Decimal("5.0575")))
        self.assertGreater(sum((p * q for p, q in zip(forecast, result.power_mw)), Decimal(0)), 0)

    def test_inefficient_30_to_40_round_trip_is_skipped(self) -> None:
        result = optimize_plan((Decimal(30), Decimal(40)) + (Decimal(0),) * 22, Decimal("0.05"))
        self.assertEqual(result.power_mw, (Decimal(0),) * 24)

    def test_marginal_sale_does_not_freeze_a_loss_after_formatting(self) -> None:
        marginal_sale_price = Decimal(1) / Decimal("0.7225") + Decimal("0.0000003")
        forecast = (Decimal(1),) * 3 + (marginal_sale_price,) * 2 + (Decimal(0),) * 19
        result = optimize_plan(forecast, Decimal("0.05"))
        self.assertEqual(result.status, "optimal")
        assert result.power_mw is not None
        self.assertGreaterEqual(
            sum((price * power for price, power in zip(forecast, result.power_mw)), Decimal(0)),
            Decimal(0),
        )

    def test_negative_forecast_revenue_from_candidate_prefers_zero_plan(self) -> None:
        marginal_sale_price = Decimal(1) / Decimal("0.7225") + Decimal("0.0000003")
        forecast = (Decimal(1),) * 3 + (marginal_sale_price,) * 2 + (Decimal(0),) * 19
        history = curve(DAY - timedelta(days=7), forecast)
        losing_plan = (
            Decimal("-6.5386"), Decimal(-7), Decimal(0), Decimal(7),
            Decimal("2.7816"),
        ) + (Decimal(0),) * 19
        self.assertLess(sum((price * power for price, power in zip(forecast, losing_plan)), Decimal(0)), 0)
        frozen = freeze_decision(
            DAY, history, Decimal("0.05"),
            optimizer=lambda _prices, _soc: OptimizationResult(losing_plan, "optimal", None, Decimal(1)),
        )
        self.assertEqual(frozen.power_mw, (Decimal(0),) * 24)
        self.assertEqual(frozen.predicted_revenue, Decimal(0))

    def test_negative_price_charges_without_simultaneous_discharge(self) -> None:
        forecast = (Decimal(-100),) + (Decimal(0),) * 23
        result = optimize_plan(forecast, Decimal("0.05"))
        self.assertEqual(result.status, "optimal")
        assert result.power_mw is not None
        self.assertEqual(result.power_mw[0], Decimal(-7))
        self.assertEqual(result.power_mw[1:], (Decimal(0),) * 23)
        self.assertEqual(validate_plan(result.power_mw, Decimal("0.05")).ending_soc, Decimal("0.475"))

    def test_full_battery_cannot_charge_at_negative_price(self) -> None:
        forecast = (Decimal(-100),) + (Decimal(0),) * 23
        result = optimize_plan(forecast, Decimal("0.95"))
        self.assertEqual(result.power_mw, (Decimal(0),) * 24)

    def test_last_hour_charge_is_kept_as_next_day_soc(self) -> None:
        frozen = freeze_decision(
            DAY,
            curve(DAY - timedelta(days=7), (0,) * 23 + (-100,)),
            Decimal("0.05"),
        )
        validation = validate_plan(frozen.power_mw, Decimal("0.05"))
        self.assertEqual(frozen.power_mw, (Decimal(0),) * 23 + (Decimal(-7),))
        self.assertEqual(validation.hourly_ending_soc[-1], Decimal("0.475"))
        self.assertEqual(validation.ending_soc, Decimal("0.475"))

    def test_multiple_price_swings_still_freeze_a_trading_plan(self) -> None:
        forecast = (
            -40, 51, 39, -67, -6, 134, 54, 21, 60, 48, -84, 55,
            -97, 132, 114, 20, -34, 41, -41, -51, 83, 20, 38, 114,
        )
        history = curve(DAY - timedelta(days=7), forecast)
        frozen = freeze_decision(DAY, history, Decimal("0.05"))
        self.assertEqual(frozen.solver_status, "optimal")
        self.assertIsNone(frozen.fallback_reason)
        self.assertNotEqual(frozen.power_mw, (Decimal(0),) * 24)
        self.assertEqual(validate_plan(frozen.power_mw, Decimal("0.05")).ending_soc, frozen.ending_soc)

    def test_freeze_then_settle_uses_actual_prices_only_for_revenue(self) -> None:
        forecast = (-100,) + (0,) * 23
        history = curve(DAY - timedelta(days=7), forecast)
        frozen = freeze_decision(DAY, history, Decimal("0.05"))
        first = settle_day(frozen, curve(DAY, (10,) + (0,) * 23))
        second = settle_day(frozen, curve(DAY, (-20,) + (0,) * 23))
        self.assertEqual(first.decision, second.decision)
        self.assertEqual(frozen.power_mw[0], Decimal(-7))
        self.assertEqual(first.daily_revenue, Decimal(-70))
        self.assertEqual(second.daily_revenue, Decimal(140))
        self.assertEqual(first.next_starting_soc, Decimal("0.475"))

    def test_solver_exception_returns_checked_zero_plan_and_reason(self) -> None:
        def broken(_prices: tuple[Decimal, ...], _soc: Decimal) -> OptimizationResult:
            raise RuntimeError("solver crashed")

        frozen = freeze_decision(
            DAY, curve(DAY - timedelta(days=7), (0,) * 24),
            Decimal("0.05"), optimizer=broken,
        )
        self.assertEqual(frozen.power_mw, (Decimal(0),) * 24)
        self.assertEqual(frozen.ending_soc, Decimal("0.05"))
        self.assertEqual(frozen.solver_status, "error")
        assert frozen.fallback_reason is not None
        self.assertIn("solver crashed", frozen.fallback_reason)
        self.assertTrue(frozen.plan_valid)

    def test_timeout_and_invalid_rounded_candidate_return_zero_plan(self) -> None:
        history = curve(DAY - timedelta(days=7), (0,) * 24)
        for result, expected_status in (
            (OptimizationResult(None, "time_limit", "limit reached", None), "time_limit"),
            (OptimizationResult(None, "no_solution", "no feasible incumbent", None), "no_solution"),
            (OptimizationResult((Decimal("7.00011"),) + (Decimal(0),) * 23,
                                "optimal", None, Decimal(1)), "optimal"),
        ):
            with self.subTest(status=expected_status, candidate=result.power_mw):
                frozen = freeze_decision(DAY, history, Decimal("0.05"),
                                         optimizer=lambda _prices, _soc: result)
                self.assertEqual(frozen.power_mw, (Decimal(0),) * 24)
                self.assertEqual(frozen.solver_status, expected_status)
                self.assertIsNotNone(frozen.fallback_reason)
                self.assertTrue(frozen.plan_valid)

    def test_last_hour_precision_overflow_falls_back_at_upper_soc(self) -> None:
        history = curve(DAY - timedelta(days=7), (0,) * 24)
        invalid = OptimizationResult(
            (Decimal(0),) * 23 + (Decimal("-0.00011"),), "optimal", None, Decimal(0)
        )
        frozen = freeze_decision(
            DAY, history, Decimal("0.95"),
            optimizer=lambda _prices, _soc: invalid,
        )
        self.assertEqual(frozen.power_mw, (Decimal(0),) * 24)
        self.assertEqual(frozen.ending_soc, Decimal("0.95"))
        self.assertIn("period 24", frozen.fallback_reason or "")

    def test_submission_precision_can_be_set_per_decision(self) -> None:
        history = curve(DAY - timedelta(days=7), (-100,) + (0,) * 23)
        frozen = freeze_decision(
            DAY, history, Decimal("0.05"), power_decimal_places=2
        )
        self.assertEqual(frozen.power_decimal_places, 2)
        self.assertEqual(frozen.power_mw[0], Decimal(-7))
        self.assertTrue(all(power == power.quantize(Decimal("0.01")) for power in frozen.power_mw))

    def test_optimizer_chooses_best_plan_at_submission_precision(self) -> None:
        forecast = (Decimal(30), Decimal(50)) + (Decimal(0),) * 22
        result = optimize_plan(forecast, Decimal("0.05"), power_decimal_places=2)
        self.assertEqual(result.status, "optimal")
        assert result.power_mw is not None
        self.assertEqual(result.power_mw, (Decimal("-6.99"), Decimal("5.05")) + (Decimal(0),) * 22)
        self.assertGreaterEqual(validate_plan(result.power_mw, Decimal("0.05"), 2).ending_soc, Decimal("0.05"))

    def test_solver_keeps_primary_revenue_above_feasible_witness(self) -> None:
        forecast = tuple(map(Decimal, (2, 9, -50, 14, 41, 71, 22, 33))) + (Decimal(0),) * 16
        witness = tuple(map(Decimal, (
            "-7", "-0.8235", "-7", "0", "3.7099", "7", "-6.9999", "5.0575",
        ))) + (Decimal(0),) * 16
        validate_plan(witness, Decimal("0.05"))
        witness_revenue = sum((price * power for price, power in zip(forecast, witness)), Decimal(0))
        self.assertEqual(witness_revenue, Decimal("990.5941"))
        result = optimize_plan(forecast, Decimal("0.05"))
        self.assertEqual(result.status, "optimal")
        assert result.power_mw is not None
        self.assertGreaterEqual(
            sum((price * power for price, power in zip(forecast, result.power_mw)), Decimal(0)),
            witness_revenue,
        )

    def test_validator_checks_every_hour_and_formatted_precision(self) -> None:
        with self.assertRaisesRegex(ValueError, "period 24"):
            validate_plan((Decimal(0),) * 23 + (Decimal("-0.0001"),), Decimal("0.95"))
        with self.assertRaisesRegex(ValueError, "precision"):
            validate_plan((Decimal("0.00001"),) + (Decimal(0),) * 23, Decimal("0.05"))
        self.assertEqual(len(validate_plan((Decimal(0),) * 24, Decimal("0.05")).hourly_ending_soc), 24)


if __name__ == "__main__":
    unittest.main()
