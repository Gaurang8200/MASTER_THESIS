# MO_Changes
from __future__ import annotations

from collections.abc import Callable


OutputCallback = Callable[[str], None]


def emit_method_execution(
    method_name: str,
    output_callback: OutputCallback | None = None,
) -> str:
    message = f"DEBUG METHOD: Executing {method_name}"
    print(message)
    if output_callback is not None and output_callback is not print:
        output_callback(message)
    return message
