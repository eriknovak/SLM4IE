"""Task-family converter backends.

Importing this package registers every converter with the driver's registry
through the `@register_converter` decorator. Import it for its side effects
before calling `slm4ie.data.tasks.driver.get_converter`.
"""

from slm4ie.data.tasks.converters import (  # noqa: F401
    sentiment,
    spans,
    superglue,
)
