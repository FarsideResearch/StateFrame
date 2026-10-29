"""Price helpers for the shop checkout."""


def apply_discount(price: float, percent: float) -> float:
    """Return the price after taking ``percent`` percent off, rounded to cents."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * percent / 100, 2)


def order_total(prices: list[float], percent: float = 0) -> float:
    """Sum the line prices, then apply one order-wide discount."""
    return apply_discount(sum(prices), percent)
