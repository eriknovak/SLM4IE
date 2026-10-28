"""Small statistics helpers shared by experiment analyses."""

import math
from collections.abc import Sequence


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a share.

    The Wilson interval is used rather than the textbook normal one because
    shares near 0 or 1 would push the normal interval past the ends of the
    scale.

    Args:
        successes: Count of items with the property.
        total: Count of items.
        z: Standard-normal quantile; the default is the 95% interval.

    Returns:
        The interval's lower and upper bounds, or `(0.0, 0.0)` when `total`
        is zero.
    """
    if total == 0:
        return 0.0, 0.0

    # compute the interval centre and spread
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> tuple[float, float]:
    """Return Cohen's kappa and raw agreement for two raters' yes-or-no calls.

    Args:
        pairs: One (first rater, second rater) pair per item.

    Returns:
        Kappa and the share of items both raters called alike. Kappa is 1.0
        when both raters give every item the same call, where chance agreement
        is already complete.

    Raises:
        ValueError: If `pairs` is empty.
    """
    if not pairs:
        raise ValueError("cohen_kappa needs at least one pair")

    # compute observed and chance agreement
    total = len(pairs)
    observed = sum(first == second for first, second in pairs) / total
    first_yes = sum(first for first, _ in pairs) / total
    second_yes = sum(second for _, second in pairs) / total
    expected = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)

    # kappa is 1.0 when chance agreement is already complete
    return ((observed - expected) / (1 - expected) if expected < 1 else 1.0), observed
