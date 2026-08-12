"""Task-family converter backends.

Importing this package registers every converter with the driver's registry
through the `@register_converter` decorator. Import it for its side effects
before calling `slm4ie.data.task_converter.get_converter`.
"""

from slm4ie.data.task_converters import (  # noqa: F401
    sentiment,
    spans,
    superglue,
)
