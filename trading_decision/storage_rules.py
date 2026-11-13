"""Shared storage limits and submission-format defaults."""

from decimal import Decimal


HOURS = 24
CAPACITY_MWH = Decimal("14")
MIN_SOC = Decimal("0.05")
MAX_SOC = Decimal("0.95")
MAX_POWER_MW = Decimal("7")
CHARGE_EFFICIENCY = Decimal("0.85")
DISCHARGE_EFFICIENCY = Decimal("0.85")
DEFAULT_POWER_DECIMAL_PLACES = 4


def power_step(decimal_places: int) -> Decimal:
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int) or not 0 <= decimal_places <= 9:
        raise ValueError("power decimal places must be an integer in 0..9")
    return Decimal(1).scaleb(-decimal_places)
