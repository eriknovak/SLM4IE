"""Small statistics helpers shared by experiment analyses."""

import math
from typing import Tuple


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
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
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)
